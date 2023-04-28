from hashlib import md5
import os
import copy

import torch
import torch.nn.functional as F
from torch.nn.utils import prune
import torchvision

import torch_pruning as tp

from models.resnet import BasicBlock, PaddingLayer, PaddingLayerPruner
from torchvision.models.efficientnet import MBConv

from params import CACHE_DIR

@torch.no_grad()
def get_all_predictions(dataset, model, batch_size=256, num_workers=6, cache=True, raw=False):
    if cache:
        cache_id = md5(
          (str([p for p in model.parameters()]) + dataset.__class__.__name__).encode()
        ).hexdigest()
        cache_path = os.path.join(CACHE_DIR, '{}{}.pt'.format(cache_id, int(raw)))

        if os.path.exists(cache_path):
            print('Loading cached predictions from', cache_path)
            return torch.load(cache_path)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=False)

    labels = []
    predictions = []

    model.eval()
    for i, batch in enumerate(loader):
        x, y = batch[0].to(model.device), batch[1]
        output = model(x, raw).cpu()

        labels.append(y)
        predictions.append(output)

    labels = torch.cat(labels)
    predictions = torch.cat(predictions)

    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save((labels, predictions), cache_path)

    return labels, predictions

class SoftDataset(torch.utils.data.Dataset):
    def __init__(self, parent_dataset, model=None, weights=None, batch_size=256,
            apply_fix_labels=True, data_is_deterministic=False):

        if model is None:
            if data_is_deterministic:
                self.weights = weights
                self.weight_fn = None
            else:
                self.weights = None
                self.weight_fn = weights
            self.labels = None
        elif data_is_deterministic:
            labels, predictions = get_all_predictions(parent_dataset, model)
            self.weights = weights
            self.weight_fn = None
            self.labels = fix_labels(predictions, labels) if apply_fix_labels else predictions
        else:
            self.original_model = copy.deepcopy(model)
            self.original_model.eval()
            self.weights = None
            self.weight_fn = weights
            self.labels = None
        
        self.parent_dataset = parent_dataset
        self.fix_labels = apply_fix_labels

        self.attribute_index_present = len(parent_dataset[0]) > 3
    
    def __len__(self):
        return len(self.parent_dataset)
    
    def __getitem__(self, idx):
        weight = self.weights[idx] if self.weights is not None else 1

        if self.attribute_index_present:
            if self.labels is None:
                return (self.parent_dataset[idx][0],
                        self.parent_dataset[idx][1],
                        weight,
                        self.parent_dataset[idx][3])
            return (self.parent_dataset[idx][0],
                    self.labels[idx],
                    weight,
                    self.parent_dataset[idx][3])
        else:
            if self.labels is None:
                return (*self.parent_dataset[idx], weight)
            return self.parent_dataset[idx][0], self.labels[idx], weight

@torch.no_grad()
def fix_labels(output, labels):
    num_classes = output.shape[1]
    if num_classes == 1:
        is_correct = (output.round() == labels).float()
        return output*is_correct + labels*(1-is_correct)
    else:
        is_correct = (output.argmax(1) == labels).float().unsqueeze(1)
        return output*is_correct + F.one_hot(labels, num_classes=num_classes)*(1-is_correct)

class PrunableConv2d(torch.nn.Module):
    LAST_PARAMS = [('include_flags', 'last_if')]

    def __init__(self, parent_conv):
        super().__init__()
        device = parent_conv.weight.device

        include_flags = torch.ones(parent_conv.weight.shape[0], device=device)
        self.include_flags = torch.nn.parameter.Parameter(include_flags, requires_grad=False)

        self.parent_conv = parent_conv
        self.batch_norm = None
        self.flop_const = None

    def forward(self, x):
        x = self.parent_conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        return x * self.include_flags.view(-1, 1, 1)

    def get_mask(self):
        target_shape = self.parent_conv.weight.shape
        return self.include_flags[:, None, None, None].expand(target_shape).int()

    def get_flops(self):
        if self.parent_conv.groups == 1:
            if self.last_if is None:
                in_channels = self.parent_conv.in_channels
            else:
                in_channels = self.last_if.sum()
            out_channels = self.include_flags.sum()

            return in_channels * self.flop_const * out_channels / self.parent_conv.groups

        if self.last_if is None:
            in_channels = torch.tensor(self.parent_conv.in_channels)
        else:
            in_channels = self.last_if
        
        in_channels = in_channels.view(-1, self.parent_conv.groups).sum(0) \
            .repeat(self.parent_conv.out_channels // self.parent_conv.groups)
        out_channels = self.include_flags

        return (in_channels * out_channels).sum() * self.flop_const

    def prune(self):
        mask = self.get_mask()
        prune.custom_from_mask(self.parent_conv, 'weight', mask)
        if self.parent_conv.bias is not None:
            prune.custom_from_mask(self.parent_conv, 'bias', self.include_flags)

    @classmethod
    def register_flops(cls, model, input_shape):
        # If there were linear layers within the model, would need to check for them
        modules = [module for module in model.modules() if isinstance(module, cls)]

        for i in range(len(modules)):
            for param, param_dest in cls.LAST_PARAMS:
                value = getattr(modules[i-1], param) if i > 0 else None
                setattr(modules[i], param_dest, value)

        # TODO: Clean up (Do this in a better way)
        for module in model.modules():
            if isinstance(module, (BasicBlock, MBConv)):
                for param, param_dest in cls.LAST_PARAMS:
                    setattr(modules[i], param_dest, None)

        def flop(module, inp, outp):
            module.flop_const = outp.shape[-1] * outp.shape[-2] * \
                module.parent_conv.kernel_size[0] *  module.parent_conv.kernel_size[1]
        hooks = [module.register_forward_hook(flop) for module in modules]

        model(torch.ones(1, *input_shape, device=model.device))
        for hook in hooks:
            hook.remove()

class Conv2dBN(PrunableConv2d):
    LAST_PARAMS = [('include_flags', 'last_if'),
                   ('bottleneck',    'last_bn')]

    def __init__(self, parent_conv, beta=1, init_bn=0):
        super().__init__(parent_conv)
        
        bottleneck = init_bn * torch.ones(parent_conv.weight.shape[0],
            device=parent_conv.weight.device, requires_grad=True)
        self.bottleneck = torch.nn.parameter.Parameter(bottleneck)

        self.beta = beta
        self.last_bn = None
        self.threshold = None

        self.batch_norm = None
        
    def forward(self, x):
        if self.threshold is None:
            bn = (self.beta * self.bottleneck.view(-1, 1, 1)).sigmoid()
        else:
            bn = ((self.beta * self.bottleneck.view(-1, 1, 1)).sigmoid() > self.threshold).float()

        x = self.parent_conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        return x * bn
    
    def get_flops(self, use_bottlenecks=True):
        if not use_bottlenecks:
            return super().get_flops()

        if self.parent_conv.groups == 1:
            if self.last_bn is None:
                in_channels = self.parent_conv.in_channels
            else:
                in_channels = torch.sigmoid(self.beta * self.last_bn).sum()

            out_channels = torch.sigmoid(self.beta * self.bottleneck).sum()
            flops = in_channels * self.flop_const * out_channels
            min_flops = in_channels * self.flop_const if self.last_bn is None else self.flop_const
        else:
            if self.last_bn is None:
                in_channels = torch.tensor(self.parent_conv.in_channels)
            else:
                in_channels = torch.sigmoid(self.beta * self.last_bn)
            
            in_channels = in_channels.view(-1, self.parent_conv.groups).sum(0) \
                .repeat(self.parent_conv.out_channels // self.parent_conv.groups)
            out_channels = torch.sigmoid(self.beta * self.bottleneck)
            flops = (in_channels * out_channels).sum() * self.flop_const
            min_flops = in_channels // self.parent_conv.groups * self.flop_const \
                if self.last_bn is None else self.flop_const

        return max(flops, min_flops)

    def get_threshold_flops(self, threshold):
        in_channel_count = (torch.sigmoid(self.beta * self.last_bn) > threshold).float().sum() \
                           if self.last_bn is not None \
                           else self.parent_conv.in_channels

        out_channels = (torch.sigmoid(self.beta * self.bottleneck) > threshold).float()

        in_channel_count = max(in_channel_count, 1)
        if out_channels.sum() < 1:
            out_channels[self.bottleneck.argmax()] = 1

        return in_channel_count * self.flop_const * out_channels

@torch.no_grad()
def replace_conv2_layers(module, new_class, **kwargs):
    if isinstance(module, torch.nn.Conv2d) and module.groups == 1:
        return new_class(module, **kwargs).to(module.weight.device)
    else:
        last_conv = None
        for name, submodule in module.named_children():
            masked_conv = replace_conv2_layers(submodule, new_class, **kwargs)
            if masked_conv is not None:
                setattr(module, name, masked_conv)
            elif isinstance(submodule, torch.nn.BatchNorm2d) and last_conv is not None:
                last_conv.batch_norm = submodule
                setattr(module, name, torch.nn.Identity())
            last_conv = masked_conv
        return None
    
@torch.no_grad()
def revert_conv2_layers(module, old_class):
    if isinstance(module, old_class):
        return module
    else:
        last_mod = None
        for name, submodule in module.named_children():
            mod_to_revert = revert_conv2_layers(submodule, old_class)
            if mod_to_revert is not None:
                setattr(module, name, mod_to_revert.parent_conv)
            elif last_mod is not None and isinstance(submodule, torch.nn.Identity):
                setattr(module, name, last_mod.batch_norm)
            last_mod = mod_to_revert

        return None

def argsort_convs_by_scores(convs, scores):
    inds = sum([
         [(i, j) for j in range(len(convs[i].include_flags))]
         for i in range(len(convs))
    ], [])
    return torch.tensor(inds)[torch.cat(scores).argsort()]

def true_prune(model, preprune_state_dict, input_shape):
    model.eval()

    convs = [module for module in model.modules()
             if isinstance(module, (torch.nn.Conv2d))]
    norms = [module for module in model.modules()
             if isinstance(module, (torch.nn.BatchNorm2d,
                                    torch.nn.LayerNorm))]

    for module in convs:
        if hasattr(module, 'weight_mask'):
            prune.remove(module, name='weight')
            if module.bias is not None:
                prune.remove(module, name='bias')

    ex_inp = 100*torch.ones(1, *input_shape, device=model.device)

    DG = tp.DependencyGraph()
    DG.register_customized_layer(PaddingLayer, PaddingLayerPruner)
    DG.build_dependency(model, example_inputs=ex_inp)
    pruning_plan = tp.DependencyGroup()

    def hook(module, inp, outp):
        nonlocal pruning_plan

        if isinstance(module, torch.nn.Linear):
            axes = tuple(range(len(inp[0].shape) - 1))
            pruning_idxs = (inp[0].sum(axis=axes) == 0).nonzero().view(-1).tolist()
            module_plan = DG.get_pruning_plan(module, tp.prune_linear_in_channels, idxs=pruning_idxs)
        elif isinstance(module, torch.nn.Conv2d):
            pruning_idxs = (inp[0].sum(axis=(0,2,3)) == 0).nonzero().view(-1).tolist()
            module_plan = DG.get_pruning_plan(module, tp.prune_conv_in_channels, idxs=pruning_idxs)

        for dep, idxs in module_plan.items:
            pruning_plan.add_and_merge(dep, idxs)

        return outp.abs()

    def norm_hook(module, inp, outp):
        return inp[0]

    handles = [c.register_forward_hook(hook) for c in convs]
    handles += [n.register_forward_hook(norm_hook) for n in norms]
    model(ex_inp)

    for h in handles:
        h.remove()

    # Restore weights from original model to avoid leftover unpruned zeros
    if preprune_state_dict is not None:
        model.load_state_dict(preprune_state_dict)

    device = model.device
    model.cpu()

    pruning_plan.exec()
    model.to(device)

    model.zero_grad(set_to_none=True)
