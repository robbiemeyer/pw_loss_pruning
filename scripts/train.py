import os
import sys
import pandas as pd
import torch

from params import LOG_PATH, CHECKPOINT_PATH

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BackboneFinetuning, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import models
import datasets

name = 'cifar10_1_shortcut_98'

train_data, test_datas = datasets.get_cifar10_data(validation=False, imagenet_size=False, mode='shortcut')
#val_data = torch.utils.data.ConcatDataset(test_datas.values())
val_data = test_datas['none']

run_name = name + '_' + sys.argv[1] if len(sys.argv) > 1 else name

model = models.CIFAR10Model()
# Cifar1 - 0.1
backbone_finetuning = BackboneFinetuning(0, backbone_initial_ratio_lr=0.1, verbose=False)

checkpoint_path = os.path.join(CHECKPOINT_PATH, run_name)
model_checkpoint = ModelCheckpoint(dirpath=checkpoint_path, filename=run_name + '_{val_loss:.2f}',
                                   monitor='val_loss', save_last=True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True, num_workers=6)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=256, shuffle=False, num_workers=6)

trainer = Trainer(
    max_epochs=model.num_epochs,
    accelerator='gpu',
    devices=[0],
    logger=TensorBoardLogger(LOG_PATH, name=run_name),
    callbacks=[backbone_finetuning, model_checkpoint],
    enable_progress_bar=True
)

trainer.fit(model, train_loader, val_loader)
