from copy import deepcopy

import torch
from torch.nn.utils import prune

from prune.utils import true_prune

def one_shot_prune(model, target_reduction, pruner, retrainer, train_loader, input_shape,
        restore_weights=True):
    print('Pruning...')
    preprune_state_dict = deepcopy(model.state_dict()) if restore_weights else None

    pruner.prune(model, target_reduction)

    true_prune(model, preprune_state_dict, input_shape)

    print('Retraining...')
    retrainer.fit(model, train_loader() if callable(train_loader) else train_loader)
