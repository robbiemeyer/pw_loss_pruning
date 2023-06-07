import numpy as np
import pandas as pd
import argparse
import os
from glob import glob

import torch
import torchvision

import models
from prune.pruner import RandomPrune, AutoBot, ScoreWeightedAutoBot, \
        TaylorPrune, ScoreWeightedTaylorPrune
from prune.utils import SoftDataset, sw_weights
from prune.retrainer import GenericRetrainer
from prune.strategy import one_shot_prune
from prune.evaluate import evaluate_model
from params import *
from datasets import get_celebA_data, get_cifar10_data, get_fitzpatrick17k_data

torch.multiprocessing.set_sharing_strategy('file_system')

# Read Arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
        choices=['celeba_1', 'celeba_2', 'cifar10_1', 'cifar10_2', 'fitz17k_1', 'fitz17k_2',
                 'celeba_1_alt1', 'celeba_1_alt2', 'celeba_1_alt3', 'cifar10_1_imbal',
                 'cifar10_1_shortcut'])
    parser.add_argument('--gpu_id', type=int, default=0,
        help='The id of the GPU to train on.')
    parser.add_argument('--output', required=True,
        help='The filename to output results to')
    parser.add_argument('--target_reduction', type=float, required=True,
        help='The proportion of parameters to remove')
    parser.add_argument('--pruned_model_output', default=None)

    parser.add_argument('--method', default='random',
        choices=['magnitude', 'random', 'autobot', 'sw_autobot', 'taylor', 'sw_taylor'])
    parser.add_argument('--validation', action='store_true')

    # Score Weighted Params
    parser.add_argument('--sw_gamma', type=float, default=5)
    parser.add_argument('--sw_min_weight', type=float, default=0.75)

    # AutoBot Params
    parser.add_argument('--ab_beta', type=float, default=1)
    parser.add_argument('--ab_gamma', type=float, default=1e-3)
    parser.add_argument('--ab_iters', type=int, default=200)
    parser.add_argument('--ab_lr', type=float, default=1e-3)

    # Taylor/Ide Params
    parser.add_argument('--tay_batches', type=float, default=16)
    parser.add_argument('--tay_prunes_per_iter', type=int, default=1)
    parser.add_argument('--tay_lr', type=float, default=1e-3)
    parser.add_argument('--tay_ckpt_name')

    # Other params
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_prunes', type=int, default=10)
    parser.add_argument('--epochs_per_prune', type=int, default=5)
    parser.add_argument('--comment', default=' ')
    parser.add_argument('--fix_labels', type=int, default=-1,
        choices=[-1, 0, 1]) # -1 = unset, 0 = false, 1 = true
    parser.add_argument('--log_name', default=None)
    parser.add_argument('--use_soft_retrain', type=int, default=-1,
        choices=[-1, 0, 1]) # -1 = unset, 0 = false, 1 = true
    parser.add_argument('--use_weights_retrain', type=int, default=-1,
        choices=[-1, 0, 1]) # -1 = unset, 0 = false, 1 = true

    return parser.parse_args()

if __name__ == "__main__" :
    args = parse_args()

    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    # Load data/model/related params
    if args.model in ('celeba_1', 'celeba_1_alt1', 'celeba_1_alt2', 'celeba_1_alt3'):
        if args.model == 'celeba_1':
            train_data, test_datas = get_celebA_data(validation=args.validation)
            model = models.CelebAModel.load_from_checkpoint(CELEBA_MODEL_1_CKPT)
        elif args.model == 'celeba_1_alt1':
            train_data, test_datas = get_celebA_data(validation=args.validation, alpha=1, beta=1)
            model = models.CelebAModel.load_from_checkpoint(CELEBA_MODEL_1_ALT1_CKPT)
        elif args.model == 'celeba_1_alt2':
            train_data, test_datas = get_celebA_data(validation=args.validation, alpha=5, beta=1)
            model = models.CelebAModel.load_from_checkpoint(CELEBA_MODEL_1_ALT2_CKPT)
        elif args.model == 'celeba_1_alt3':
            train_data, test_datas = get_celebA_data(validation=args.validation, alpha=1, beta=5)
            model = models.CelebAModel.load_from_checkpoint(CELEBA_MODEL_1_ALT3_CKPT)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
        lr = 1e-4
        batch_size = 256
        num_epochs = 30
        data_is_deterministic = True
    elif args.model == 'celeba_2':
        train_data, test_datas = get_celebA_data(validation=args.validation)
        model = models.CelebAModel2.load_from_checkpoint(CELEBA_MODEL_2_CKPT)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
        lr = 5e-4
        batch_size = 64
        num_epochs = 10
        data_is_deterministic = True
    elif args.model in ('cifar10_1', 'cifar10_1_imbal', 'cifar10_1_shortcut'):
        if args.model == 'cifar10_1_imbal':
            mode = 'imbalanced'
        elif args.model == 'cifar10_1_shortcut':
            mode = 'shortcut'
        else:
            mode = None
        train_data, test_datas = get_cifar10_data(validation=args.validation, mode=mode)

        if args.model == 'cifar10_1_imbal':
            model = models.CIFAR10Model.load_from_checkpoint(CIFAR10_MODEL_1_IMBAL_CKPT)
        elif args.model == 'cifar10_1_shortcut':
            model = models.CIFAR10Model.load_from_checkpoint(CIFAR10_MODEL_1_SHORTCUT_CKPT)
        elif args.validation:
            model = models.CIFAR10Model.load_from_checkpoint(CIFAR10_MODEL_1_VAL_CKPT)
        else:
            model = models.CIFAR10Model.load_from_checkpoint(CIFAR10_MODEL_1_CKPT)

        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        lr = 1e-3
        batch_size = 256
        num_epochs = 200
        data_is_deterministic = False
    elif args.model == 'cifar10_2':
        train_data, test_datas = get_cifar10_data(validation=args.validation, imagenet_size=True)
        if args.validation:
            model = models.CIFAR10Model2.load_from_checkpoint(CIFAR10_MODEL_2_VAL_CKPT)
        else:
            model = models.CIFAR10Model2.load_from_checkpoint(CIFAR10_MODEL_2_CKPT)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        lr = 1e-5
        batch_size = 128
        num_epochs = 10
        data_is_deterministic = False
    elif args.model == 'fitz17k_1':
        train_data, test_datas = get_fitzpatrick17k_data(validation=args.validation)
        model = models.FitzModel.load_from_checkpoint(FITZ17k_MODEL_1_CKPT)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        lr = 1e-4
        batch_size = 64
        num_epochs = 30
        data_is_deterministic = True
    elif args.model == 'fitz17k_2':
        train_data, test_datas = get_fitzpatrick17k_data(validation=args.validation)
        model = models.FitzModel2.load_from_checkpoint(FITZ17k_MODEL_2_CKPT)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        lr = 1e-5
        batch_size = 32
        num_epochs = 50
        data_is_deterministic = True

    model.to(device)

    input_shape = train_data[0][0].shape
    num_workers = 6

    log_dir = os.path.join(LOG_PATH, args.log_name, args.method + ' ' + args.comment) \
            if args.log_name is not None else None

    # Get pruner
    if args.method == 'random':
        pruner = RandomPrune(input_shape)
    elif args.method == 'autobot':
        pruner = AutoBot(train_data, args.ab_beta, args.ab_gamma, batch_size=args.batch_size,
            num_workers=num_workers,
            fix_labels=args.fix_labels > 0 if args.fix_labels != -1 else False,
            lr=args.ab_lr, num_iters=args.ab_iters, data_is_deterministic=data_is_deterministic,
            log_dir=log_dir)
    elif args.method == 'sw_autobot':
        pruner = ScoreWeightedAutoBot(train_data, args.sw_gamma, args.sw_min_weight, args.ab_beta,
            args.ab_gamma, batch_size=args.batch_size, lr=args.ab_lr, num_iters=args.ab_iters,
            fix_labels=args.fix_labels > 0 if args.fix_labels != -1 else True,
            data_is_deterministic=data_is_deterministic, log_dir=log_dir)
    elif args.method in ('taylor', 'sw_taylor'):
        if args.tay_ckpt_name is not None:
            save_ckpt = os.path.join(CACHE_DIR,
                '{}_{}.ckpt'.format(args.target_reduction, args.tay_ckpt_name))
            old_ckpts = sorted(glob('{}/*_{}.ckpt'.format(CACHE_DIR, args.tay_ckpt_name)),
                               key=lambda x: float(x.split('/')[-1].split('_')[0]),
                               reverse=True)
            load_ckpt = old_ckpts[0] if len(old_ckpts) > 0 else None
        else:
            save_ckpt = None
            load_ckpt = None
        if args.method == 'taylor':
            pruner = TaylorPrune(train_data, args.tay_batches, args.tay_prunes_per_iter,
                batch_size=args.batch_size, lr=args.tay_lr,
                label_style = 'fixed' if args.fix_labels == 1 else 'original',
                save_checkpoint=save_ckpt, load_checkpoint=load_ckpt, log_dir=log_dir)
        else:
            pruner = ScoreWeightedTaylorPrune(train_data, args.tay_batches, args.tay_prunes_per_iter,
                args.sw_gamma, args.sw_min_weight,
                batch_size=args.batch_size, lr=args.tay_lr,
                label_style ='original' if args.fix_labels == 0 else 'fixed',
                data_is_deterministic=data_is_deterministic,
                save_checkpoint=save_ckpt, load_checkpoint=load_ckpt, log_dir=log_dir)

    use_soft_retrain = args.use_soft_retrain == 1 \
            or args.use_soft_retrain == -1 and args.method in ('sw_taylor', 'sw_autobot')
    use_weights_retrain = args.use_weights_retrain == 1 \
            or args.use_weights_retrain == -1 and args.method in ('sw_taylor', 'sw_autobot')

    # Get data for retraining
    if args.method in ('sw_taylor', 'sw_autobot') and args.use_soft_retrain == -1 and args.use_weights_retrain == -1:
        train_loader = lambda: torch.utils.data.DataLoader(pruner.soft_train_data,
            batch_size=batch_size, shuffle=True, num_workers=num_workers)
    elif use_soft_retrain or use_weights_retrain:
        soft_train_data = SoftDataset(train_data, model,
                                      weights=sw_weights(args.sw_min_weight, args.sw_gamma) if use_weights_retrain else None,
                                      apply_fix_labels=use_soft_retrain,
                                      data_is_deterministic=False)
        train_loader = torch.utils.data.DataLoader(soft_train_data, batch_size=batch_size,
                                                   shuffle=True, num_workers=num_workers)
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
            num_workers=num_workers)

    #if args.method in ('sw_taylor', 'sw_autobot') and args.use_soft_retrain == 1 and args.fix_labels == 0:
    #    def train_loader():
    #        pruner.soft_train_data.labels = None
    #        return torch.utils.data.DataLoader(pruner.soft_train_data, batch_size=batch_size,
    #                                           shuffle=True, num_workers=num_workers)
    #elif args.method in ('sw_taylor', 'sw_autobot') and args.use_soft_retrain != 0:
    #    train_loader = lambda: torch.utils.data.DataLoader(pruner.soft_train_data,
    #        batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #else:
    #    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
    #        num_workers=num_workers)

    # Get retrainer
    retrainer = GenericRetrainer(lr, num_epochs)

    # Prune!
    if args.target_reduction > 0:
        restore_weights = args.method != 'taylor'
        one_shot_prune(model, args.target_reduction, pruner, retrainer, train_loader,
                       input_shape, restore_weights=restore_weights)

    # Evaluate the pruned model
    eval_on_all = args.model in ('celeba_1', 'celeba_2', 'fitz17k_1', 'fitz17k_2',
        'celeba_1_alt1', 'celeba_1_alt2', 'celeba_1_alt3')
    results = evaluate_model(model, test_datas, args.target_reduction, input_shape, eval_on_all,
        args.comment)

    mode = 'a' if os.path.exists(args.output) else 'w'
    results.to_csv(args.output, index=False, mode=mode, header=(mode == 'w'))

    print("Performance:")
    print(results.set_index('group'))

    # Save the model
    if args.pruned_model_output is not None and args.target_reduction > 0:
        model.zero_grad(set_to_none=True)
        torch.save([model.backbone, model.fc], args.pruned_model_output)
