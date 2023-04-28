import torch
from math import ceil
import os

from prune.utils import fix_labels

class GenericRetrainer:
    def __init__(self, lr, num_epochs, verbose=True):
        self.num_epochs = num_epochs
        self.lr = lr
        self.verbose = verbose
    
    def register(self, model):
        self.model = model

        model.lr = self.lr
        model.num_epochs = self.num_epochs

        [self.optimizer], [self.scheduler] = model.configure_optimizers()
        self.optimizer.add_param_group({'params': model.backbone.parameters()})

        if model.nclasses == 1:
            self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.num_epochs)

    def fit_epoch(self, train_loader, other_model=None, weight_fn=None, apply_fix_labels=False):
        self.model.train()
        for i, data in enumerate(train_loader):
            if self.verbose:
                print('Batch {}/{}'.format(i+1, len(train_loader)), end='\r')

            inputs = data[0].to(self.model.device)
            labels = data[1].to(self.model.device)
            weights = data[2].to(self.model.device) if len(data) > 2 else 1
            if other_model is not None:
                predictions = other_model(inputs)
                if weight_fn is not None:
                    weights = weight_fn(labels, predictions)
                labels = fix_labels(predictions, labels) if apply_fix_labels else predictions

            self.optimizer.zero_grad()
            outputs = self.model(inputs, True)

            loss = (self.loss_fn(outputs, labels).squeeze() * weights).mean()
            
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()

    def fit(self, model, train_loader, verbose=True):
        if hasattr(train_loader.dataset, 'original_model'):
            other_model = train_loader.dataset.original_model
            apply_fix_labels = train_loader.dataset.fix_labels
            weight_fn = train_loader.dataset.weight_fn
        else:
            other_model = None
            apply_fix_labels = False
            weight_fn = None

        self.register(model)
        for i in range(self.num_epochs):
            if self.verbose:
                term_cols = os.get_terminal_size().columns
                epoch_str = '(Epoch {}/{})'.format(i+1, self.num_epochs)
                print(' ' * (term_cols - len(epoch_str)) + epoch_str, end='\r')
            self.fit_epoch(train_loader, other_model=other_model, weight_fn=weight_fn, 
                apply_fix_labels=apply_fix_labels)
        if self.verbose:
            print()
