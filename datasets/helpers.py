from .fitzpatrick import Fitzpatrick17k

import os
import numpy as np
import pandas as pd
import json

import torch
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as transforms

from params import DATA_ROOT, CELEBA_LABEL_INDEX, FITZ_SPLIT_INDS
from .utils import DatasetWithAttribute, ClasswiseTransformDataset, RandomSquareTransform

def get_celebA_data(validation=True, alpha=None, beta=None, attach_attribute=True):
    if alpha is not None or beta is not None:
        assert alpha is not None and beta is not None

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Use blond/non-blond as the target
    target_transform = lambda x: x.squeeze()[CELEBA_LABEL_INDEX].unsqueeze(0).float()

    train_data = torchvision.datasets.CelebA(root=DATA_ROOT, split='train',
        transform=transform, target_transform=target_transform)
    test_data = torchvision.datasets.CelebA(root=DATA_ROOT, split='valid' if validation else 'test',
        transform=transform, target_transform=target_transform)

    # Segment the test data into male and non-male
    male_ind = np.argmax(['Male' == x for x in test_data.attr_names])
    is_male = test_data.attr[:, male_ind].nonzero()
    is_female = (test_data.attr[:, male_ind] == 0).nonzero()
    test_datas = {'male': Subset(test_data, is_male), 'female': Subset(test_data, is_female)}

    # Use subsets of the dataset
    # alpha controls the number of female samples included
    # beta controls the number of blond samples
    if alpha is not None and beta is not None:
        df = pd.DataFrame({
            'is_attr': train_data.attr[:, male_ind],
            'label': train_data.attr[:, CELEBA_LABEL_INDEX]
        }).sample(frac=1, random_state=0)

        counts = df.value_counts()
        min_attr, min_label = counts.idxmin()
        min_count = counts.loc[(min_attr, min_label)]

        included_samples = pd.concat([
            df.loc[(df.is_attr == min_attr) & (df.label == min_label)],
            df.loc[(df.is_attr != min_attr) & (df.label == min_label)].iloc[:alpha*min_count],
            df.loc[(df.is_attr == min_attr) & (df.label != min_label)].iloc[:beta*min_count],
            df.loc[(df.is_attr != min_attr) & (df.label != min_label)].iloc[:alpha*beta*min_count],
        ]).index

        train_data = Subset(train_data, included_samples)
    elif attach_attribute:
        # Attach the attribute information for logging purposes
        # As this is only used for one plot, we don't try to make this work with alpha/beta
        attr_list = train_data.attr[:, male_ind]
        train_data = DatasetWithAttribute(train_data, attr_list)

    return train_data, test_datas

def get_cifar10_data(validation=True, imagenet_size=False, mode=None):
    assert mode in (None, 'shortcut', 'imbalanced')

    if imagenet_size: # For use with pretrained models
        norm = transforms.Compose([
            transforms.Resize(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    transform = transforms.Compose([
        transforms.ToTensor(),
        norm
    ])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transform
    ])

    if mode == 'shortcut':
        square_transform = RandomSquareTransform(4, 32)
        train_in_transform = transforms.Compose([
            transforms.ToTensor(),
            square_transform,
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            norm
        ])
        test_with_square_transforms = transforms.Compose([
            transforms.ToTensor(),
            square_transform,
            norm
        ])

        train_data = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True)
        affected_classes = {train_data.class_to_idx['dog']}

        train_data = ClasswiseTransformDataset(train_data, train_in_transform, train_transform,
                                               affected_classes, p=0.98)
    else:
        train_data = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True,
                                                  transform=train_transform)

    if mode == 'imbalanced':
        affected_classes = {train_data.class_to_idx[x] for x in ('cat', 'dog')}
        affected_idx = [[i for i in range(len(train_data)) if train_data.targets[i] == x]
                        for x in affected_classes]

        rng = np.random.default_rng(123456789)
        discarded_idx = sum([rng.choice(x, int(0.8 * len(x))) for x in affected_idx])
        discarded_idx = set(discarded_idx)

        included_idx =  [i for i in range(len(train_data))
                        if i not in discarded_idx]
        train_data = Subset(train_data, included_idx)

    # If validation, split the training data between train and test
    if validation:
        indices = torch.randperm(len(train_data),
            generator=torch.Generator().manual_seed(123456789))
        test_size = len(train_data)//5

        test_data = {
            'none': Subset(train_data, indices[-test_size:]),
        }
        train_data = Subset(train_data, indices[:-test_size])
    else:
        test_data = {
            'none': torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, transform=transform)
        }
        if mode == 'shortcut':
            test_data['with_square'] = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False,
                                            transform=test_with_square_transforms)

    return train_data, test_data

def get_fitzpatrick17k_data(validation=True):
    transform = transforms.Compose([
        transforms.Resize(250),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data = Fitzpatrick17k(root=DATA_ROOT, fitz_values=[1,2], transform=transform)
    med_data = Fitzpatrick17k(root=DATA_ROOT, fitz_values=[3,4], transform=transform)
    dark_data = Fitzpatrick17k(root=DATA_ROOT, fitz_values=[5,6], transform=transform)

    # If we have already made the val/test split, just load it
    # Otherwise, split the data
    if os.path.exists(FITZ_SPLIT_INDS):
        with open(FITZ_SPLIT_INDS, "r") as file:
            inds = json.load(file)

        valid_med_data = Subset(med_data, inds['valid_med'])
        test_med_data = Subset(med_data, inds['test_med'])
        valid_dark_data = Subset(dark_data, inds['valid_dark'])
        test_dark_data = Subset(dark_data, inds['test_dark'])
    else:
        valid_med_size = len(med_data) // 4
        valid_dark_size = len(dark_data) // 4

        generator = torch.Generator().manual_seed(123456789)
        valid_med_data, test_med_data = torch.utils.data.random_split(med_data,
            [valid_med_size, len(med_data) - valid_med_size], generator=generator)
        valid_dark_data, test_dark_data = torch.utils.data.random_split(dark_data,
            [valid_dark_size, len(dark_data) - valid_dark_size], generator=generator)

        inds = {
            'valid_med': valid_med_data.indices,
            'test_med': test_med_data.indices,
            'valid_dark': valid_dark_data.indices,
            'test_dark': test_dark_data.indices
        }

        with open(FITZ_SPLIT_INDS, "w") as file:
            json.dump(inds, file)

    if validation:
        return train_data, {'medium': valid_med_data, 'dark': valid_dark_data}
    else:
        return train_data, {'medium': test_med_data, 'dark': test_dark_data}
