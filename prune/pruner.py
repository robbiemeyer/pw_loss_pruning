import numpy as np
from math import ceil
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.nn.utils.prune import ln_structured, custom_from_mask

from prune.utils import Conv2dBN, SoftDataset, replace_conv2_layers, revert_conv2_layers, \
    PrunableConv2d, Conv2dBN, argsort_convs_by_scores, fix_labels, get_all_predictions

from torch.utils.tensorboard import SummaryWriter

from copy import deepcopy

class RandomPrune:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def prune(self, model, target_reduction):
        device = model.device

        model.cpu()

        replace_conv2_layers(model, PrunableConv2d)

        modules_to_prune = [module for module in model.modules()
                            if isinstance(module, PrunableConv2d)]

        random_scores = [torch.rand(module.include_flags.shape) for module in modules_to_prune]
        prune_inds = argsort_convs_by_scores(modules_to_prune, random_scores)

        PrunableConv2d.register_flops(model, self.input_shape)
        get_cost = lambda: sum([module.get_flops() for module in modules_to_prune])

        total_cost = get_cost()
        target_cost = (1-target_reduction) * total_cost

        for i, j in prune_inds:
            cost = get_cost()
            print(1 - cost/total_cost, '        ', end='\r')
            if cost < target_cost:
                break

            if modules_to_prune[i].include_flags.sum() > 1:
                modules_to_prune[i].include_flags[j] = 0
        print()

        for module in modules_to_prune:
            module.prune()

        revert_conv2_layers(model, PrunableConv2d)
        model.to(device)

class AutoBot:
    get_sample_weights = None

    def __init__(self, train_data, beta=6, gamma=1e-3, epsilon=0.001, fix_labels=False, lr=0.6,
                 num_iters=200, batch_size=64, num_workers=6, data_is_deterministic=False,
                 log_dir=None):
        self.train_data = train_data
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.fix_labels = fix_labels
        self.num_iters = num_iters
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_is_deterministic = data_is_deterministic
        self.log_dir = log_dir

        if self.log_dir is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(0)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            self.data_is_deterministic = False
        else:
            self.generator = None

    def get_weights(self, model, target_reduction):
        return None

    def prune(self, model, target_reduction):
        device = model.device

        @torch.no_grad()
        def set_include_flags(modules, target_flops, acceptable_error):
            threshold = 0.5

            i = 0
            at_target_flops = False
            while not at_target_flops and i < 100:
                i += 1

                flops = sum([module.get_threshold_flops(threshold).sum() for module in modules])
                flop_diff = flops - target_flops

                print(threshold, 1-flops.item()/total_flops, end='\r')

                if flop_diff.abs() < acceptable_error:
                    at_target_flops = True
                elif flop_diff < 0:
                    threshold -= 2**-(i+1)
                else:
                    threshold += 2**-(i+1)
            print()

            for module in modules:
                include_flags = (module.bottleneck.sigmoid() > threshold).float()
                module.include_flags = torch.nn.parameter.Parameter(include_flags, requires_grad=False)
                if module.include_flags.sum() == 0:
                    module.include_flags[module.bottleneck.argmax()] = 1

        def train_bottlenecks(train_data, total_flops, target_flops):
            bn_modules = [module for module in model.modules() if isinstance(module, Conv2dBN)]

            train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size,
                    shuffle=True, num_workers=self.num_workers, generator=self.generator)

            optimizer = torch.optim.Adam([module.bottleneck for module in bn_modules], lr=self.lr)

            if model.nclasses == 1:
                ce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
            else:
                ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

            def g_loss():
                flops = sum([module.get_flops() for module in bn_modules])
                if flops > target_flops:
                    return (flops - target_flops)/(total_flops - target_flops)
                return 1 - flops/target_flops

            def h_loss():
                bottlenecks = torch.cat([module.bottleneck for module in bn_modules])
                return (bottlenecks.sigmoid() - bottlenecks.sigmoid().round()).abs().mean()

            model.train()
            other_model = train_data.original_model if not self.data_is_deterministic else None

            i = 0
            best_loss = torch.inf
            best_state_dict = None
            num_epochs = ceil(self.num_iters / len(train_loader))
            for epoch in range(num_epochs):
                for data in train_loader:
                    if i > self.num_iters:
                        break
                    i += 1

                    inputs = data[0].to(device)
                    labels = data[1].to(device)
                    weights = data[2].to(device)
                    if other_model is not None:
                        if self.log_dir is not None:
                            old_labels = labels.squeeze()

                        old_outputs = other_model(inputs)
                        if train_data.weight_fn is not None:
                            weights = train_data.weight_fn(labels, old_outputs)
                        labels = fix_labels(old_outputs, labels) if self.fix_labels else old_outputs


                    optimizer.zero_grad()
                    outputs = model(inputs, True)

                    data_losses = ce_loss(outputs, labels).squeeze() * weights

                    if self.log_dir is not None and len(data) > 3:
                        attr_values = data[3].to(device)

                        total_loss = data_losses.sum()
                        pos_attr_pos_label_loss = old_labels * data_losses * attr_values
                        neg_attr_pos_label_loss = old_labels * data_losses * (1 - attr_values)
                        pos_attr_neg_label_loss = (1 - old_labels) * data_losses * attr_values
                        neg_attr_neg_label_loss = (1 - old_labels) * data_losses * (1-attr_values)

                        overshoot = (outputs > labels).squeeze() * old_labels + \
                                (outputs < labels).squeeze() * (1 - old_labels)

                        self.writer.add_scalar("Loss/pos_attr_pos_label",
                                               pos_attr_pos_label_loss.sum()/total_loss, i)
                        self.writer.add_scalar("Loss/neg_attr_pos_label",
                                               neg_attr_pos_label_loss.sum()/total_loss, i)
                        self.writer.add_scalar("Loss/pos_attr_neg_label",
                                               pos_attr_neg_label_loss.sum()/total_loss, i)
                        self.writer.add_scalar("Loss/neg_attr_neg_label",
                                               neg_attr_neg_label_loss.sum()/total_loss, i)

                        self.writer.add_scalar("Loss/pp_overshoot",
                                               (pos_attr_pos_label_loss * overshoot).sum()/total_loss, i)
                        self.writer.add_scalar("Loss/np_overshoot",
                                               (neg_attr_pos_label_loss * overshoot).sum()/total_loss, i)
                        self.writer.add_scalar("Loss/pn_overshoot",
                                               (pos_attr_neg_label_loss * overshoot).sum()/total_loss, i)
                        self.writer.add_scalar("Loss/nn_overshoot",
                                               (neg_attr_neg_label_loss * overshoot).sum()/total_loss, i)

                    loss1 = data_losses.mean()
                    loss2 = g_loss()
                    loss3 = h_loss()
                    loss = loss1 + self.beta*loss2 + self.gamma*loss3
                    loss.backward()
                    optimizer.step()

                    if i > 0.6 * self.num_iters and loss < best_loss:
                        best_loss = loss.detach()
                        best_state_dict = deepcopy(model.state_dict())

                    print('Epoch {}, Batch {}/{}'.format(epoch, i, len(train_loader)), end='\r')
                print()

            if self.log_dir is not None:
                self.writer.close()
            model.load_state_dict(best_state_dict)

        weights = self.get_weights(model, target_reduction) if self.data_is_deterministic else \
            self.get_sample_weights
        self.soft_train_data = SoftDataset(self.train_data, model, weights=weights,
            apply_fix_labels=self.fix_labels, data_is_deterministic=self.data_is_deterministic)

        replace_conv2_layers(model, Conv2dBN, init_bn=2.1972)

        input_shape = self.train_data[0][0].shape
        Conv2dBN.register_flops(model, input_shape)
        bn_modules = [module for module in model.modules() if isinstance(module, Conv2dBN)]

        total_flops = sum([module.get_flops(use_bottlenecks=False)
                           for module in bn_modules]).detach()
        target_flops = (1 - target_reduction) * total_flops
        acceptable_error = self.epsilon * total_flops

        train_bottlenecks(self.soft_train_data, total_flops, target_flops)

        # Prune!
        set_include_flags(bn_modules, target_flops, acceptable_error)
        for module in bn_modules:
            module.prune()

        # Cleanup
        revert_conv2_layers(model, Conv2dBN)
        model.zero_grad(set_to_none=True)

class ScoreWeightedAutoBot(AutoBot):
    def __init__(self, train_data, sw_gamma=0.5, min_weight=0.4, beta=6, gamma=1e-3, epsilon=0.01,
            fix_labels=True, lr=0.01, num_iters=None, batch_size=64, num_workers=6,
            data_is_deterministic=False, log_dir=None):
        self.sw_gamma = sw_gamma
        self.min_weight = min_weight

        if num_iters is None:
            num_iters = len(train_data) // batch_size

        super().__init__(train_data, beta, gamma, epsilon, fix_labels, lr, num_iters, batch_size,
            num_workers, data_is_deterministic, log_dir)

    @torch.no_grad()
    def get_weights(self, model, target_reduction):
        if self.min_weight == 1:
            return None

        labels, predictions = get_all_predictions(self.train_data, model)

        return self.get_sample_weights(labels, predictions)

    @torch.no_grad()
    def get_sample_weights(self, labels, predictions):
        if predictions.shape[1] == 1:
            predictions, labels = predictions.view(-1), labels.view(-1)
            scores = predictions * labels + (1 - predictions) * (1 - labels)
        else:
            scores = torch.gather(predictions, 1, labels.view(-1, 1)).view(-1)

        return self.min_weight + (1-self.min_weight) * (1-scores)**self.sw_gamma

class TaylorPrune:
    data_is_deterministic = False
    get_sample_weights = None

    def __init__(self, train_data, batches_bw_prunes=16, prunes_per_iter=1, lr=0.01, batch_size=64,
            num_workers=6, label_style='original', save_checkpoint=None, load_checkpoint=None,
            log_dir=None):

        self.train_data = train_data
        self.num_workers = num_workers

        self.lr = lr
        self.batch_size = batch_size
        self.batches = batches_bw_prunes + 1
        self.prunes_per_iter = prunes_per_iter

        self.label_style = label_style

        self.save_checkpoint = save_checkpoint
        self.load_checkpoint = load_checkpoint


        self.log_dir = log_dir
        if self.log_dir is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(0)
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.generator = None

    def get_weights(self, model, target_reduction):
        return None

    def prune(self, model, target_reduction):

        @torch.no_grad()
        def clean_scores(module, scores):
            scores[module.include_flags == 0] = torch.inf
            if module.include_flags.sum() == 1:
                return scores * torch.inf
            return scores

        @torch.no_grad()
        def score_module(module):
            grads = module.parent_conv.grad_outp.unsqueeze(-1)
            xs = module.parent_conv.outp.unsqueeze(-1)
            scores = (xs.mT @ grads).abs().mean(0).view(-1)
            return clean_scores(module, scores)

        @torch.no_grad()
        def forward_hook(module, inp, outp):
            module.outp = outp.view(*outp.shape[:2], -1).detach()

        @torch.no_grad()
        def backward_hook(module, grad_inp, grad_outp):
            module.grad_outp = grad_outp[0].view(*grad_outp[0].shape[:2], -1).detach()

        def prune_filter(convs, inds, inputs, labels, weights):
            model.eval()
            outputs = model(inputs, True)

            loss = (celoss(outputs, labels).squeeze() * weights).mean()
            loss.backward()

            scores = torch.cat([score_module(c) for c in convs])

            score_inds = scores.topk(self.prunes_per_iter, largest=False).indices
            for i, j in inds[score_inds]:
                if convs[i].include_flags.sum() > 1:
                    convs[i].include_flags[j] = 0

        @torch.no_grad()
        def log_loss(data_losses, outputs, labels, old_labels, attr_values, i):
            total_loss = data_losses.sum()

            pos_attr_pos_label_loss = old_labels * data_losses * attr_values
            neg_attr_pos_label_loss = old_labels * data_losses * (1 - attr_values)
            pos_attr_neg_label_loss = (1 - old_labels) * data_losses * attr_values
            neg_attr_neg_label_loss = (1 - old_labels) * data_losses * (1-attr_values)

            overshoot = (outputs > labels).squeeze() * old_labels + \
                    (outputs < labels).squeeze() * (1 - old_labels)

            self.writer.add_scalar("Loss/pos_attr_pos_label",
                                   pos_attr_pos_label_loss.sum()/total_loss, i)
            self.writer.add_scalar("Loss/neg_attr_pos_label",
                                   neg_attr_pos_label_loss.sum()/total_loss, i)
            self.writer.add_scalar("Loss/pos_attr_neg_label",
                                   pos_attr_neg_label_loss.sum()/total_loss, i)
            self.writer.add_scalar("Loss/neg_attr_neg_label",
                                   neg_attr_neg_label_loss.sum()/total_loss, i)

            self.writer.add_scalar("Loss/pp_overshoot",
                                   (pos_attr_pos_label_loss * overshoot).sum()/total_loss, i)
            self.writer.add_scalar("Loss/np_overshoot",
                                   (neg_attr_pos_label_loss * overshoot).sum()/total_loss, i)
            self.writer.add_scalar("Loss/pn_overshoot",
                                   (pos_attr_neg_label_loss * overshoot).sum()/total_loss, i)
            self.writer.add_scalar("Loss/nn_overshoot",
                                   (neg_attr_neg_label_loss * overshoot).sum()/total_loss, i)

        # Build weighted dataset that uses model output as labels
        weights = self.get_weights(model, target_reduction) if self.data_is_deterministic else \
            self.get_sample_weights
        self.soft_train_data = SoftDataset(self.train_data,
            model=model,
            weights=weights,
            apply_fix_labels=self.label_style == 'fixed',
            data_is_deterministic=self.data_is_deterministic)
        self.train_data = self.soft_train_data

        replace_conv2_layers(model, PrunableConv2d)
        convs = [module for module in model.modules() if isinstance(module, PrunableConv2d)]
        inds = torch.tensor(sum([
             [(i, j) for j in range(len(convs[i].include_flags))]
             for i in range(len(convs))
        ], []), device=model.device)

        input_shape = self.train_data[0][0].shape
        PrunableConv2d.register_flops(model, input_shape)

        get_cost = lambda: sum([c.get_flops() for c in convs])
        total_cost = get_cost()

        if self.load_checkpoint is not None:
            for c in convs:
                c.prune() # To create prune mask attrs
            model.load_state_dict(torch.load(self.load_checkpoint))

        handles  = [c.parent_conv.register_forward_hook(forward_hook) for c in convs]
        handles += [c.parent_conv.register_full_backward_hook(backward_hook) for c in convs]

        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size,
                shuffle=True, num_workers=self.num_workers, generator=self.generator)

        if model.nclasses == 1:
            celoss = torch.nn.BCEWithLogitsLoss(reduction='none')
        else:
            celoss = torch.nn.CrossEntropyLoss(reduction='none')

        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)

        target_cost = (1-target_reduction) * total_cost

        counter = 0
        old_labels = None
        while get_cost() > target_cost:
            for data in train_loader:
                counter += 1
                inputs = data[0].to(model.device)
                labels = data[1].to(model.device)
                weights = data[2].to(model.device)
                attr_values = data[3].to(model.device) if len(data) > 3 else None
                if self.log_dir is not None:
                    old_labels = labels.squeeze()

                if not self.data_is_deterministic:
                    if self.label_style != 'original' or self.train_data.weight_fn is not None:
                        old_outputs = self.train_data.original_model(inputs)
                    if self.train_data.weight_fn is not None:
                        weights = self.train_data.weight_fn(labels, old_outputs)
                    if self.label_style != 'original':
                        labels = fix_labels(old_outputs, labels) if self.train_data.fix_labels else \
                                old_outputs

                model.train()
                optimizer.zero_grad()
                outputs = model(inputs, True)

                data_losses = celoss(outputs, labels).squeeze() * weights
                if self.log_dir is not None and attr_values is not None:
                    log_loss(data_losses, model.reduce_fn(outputs), labels, old_labels, attr_values, counter)

                loss = data_losses.mean()
                loss.backward()
                optimizer.step()

                if counter % self.batches == 0:
                    cost = get_cost()
                    print('batch:', counter, '- cost:', 1 - (cost/total_cost).item(), end='\r')
                    if cost <= target_cost:
                        break
                    prune_filter(convs, inds, inputs, labels, weights)
        print()

        if self.log_dir is not None:
            self.writer.close()

        for module in convs:
            del module.parent_conv.outp
            del module.parent_conv.grad_outp
            module.prune()

        for handle in handles:
            handle.remove()

        if self.save_checkpoint is not None:
            torch.save(model.state_dict(), self.save_checkpoint) 
        revert_conv2_layers(model, PrunableConv2d)

class ScoreWeightedTaylorPrune(TaylorPrune):
    def __init__(self, train_data, batches_bw_prunes=16, prunes_per_iter=1, gamma=1, min_weight=1,
            lr=0.01, batch_size=64, num_workers=6, label_style='fixed', data_is_deterministic=False,
            save_checkpoint=None, load_checkpoint=None, log_dir=None):

        self.gamma = gamma
        self.min_weight = min_weight
        self.data_is_deterministic = data_is_deterministic

        if log_dir is not None:
            self.data_is_deterministic = False

        super().__init__(train_data, batches_bw_prunes, prunes_per_iter, lr, batch_size,
                num_workers, label_style, save_checkpoint, load_checkpoint, log_dir)

    @torch.no_grad()
    def get_weights(self, model, target_reduction):
        if self.min_weight == 1:
            return None

        labels, predictions = get_all_predictions(self.train_data, model)

        return self.get_sample_weights(labels, predictions)

    @torch.no_grad()
    def get_sample_weights(self, labels, predictions):
        if predictions.shape[1] == 1:
            predictions, labels = predictions.view(-1), labels.view(-1)
            scores = predictions * labels + (1 - predictions) * (1 - labels)
        else:
            scores = torch.gather(predictions, 1, labels.view(-1, 1)).view(-1)

        return self.min_weight + (1-self.min_weight) * (1-scores)**self.gamma
