import torch
from torch import nn
import torchvision
from pytorch_lightning import LightningModule

from .resnet import resnet56

class BaseModel(LightningModule):
    def __init__(self, backbone, fc, nclasses, lr=1e-3, num_epochs=10, weight=None):
        super().__init__()

        self.backbone = backbone
        self.fc = fc

        self.nclasses = nclasses
        if nclasses > 1:
            self.loss_fn = nn.CrossEntropyLoss(weight=weight)
            self.pred_fn = lambda x: torch.argmax(x, dim=1)
            self.reduce_fn = lambda x: x.softmax(dim=1)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(weight=weight)
            self.pred_fn = lambda x: x > 0
            self.reduce_fn = torch.sigmoid

        self.lr = lr
        self.num_epochs = num_epochs

    def forward(self, x, raw=False):
        x = self.backbone(x)
        x = self.fc(x)
        if raw:
            return x
        return self.reduce_fn(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, True)
        loss = self.loss_fn(logits, y)

        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch):
        x, y = batch
        logits = self(x, True)

        loss = self.loss_fn(logits, y)
        accuracy = (self.pred_fn(logits) == y).float().mean()

        return loss, accuracy

    def log_stage(self, stage, loss, acc):
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.evaluate(batch)
        self.log_stage("val", loss, accuracy)

    def test_step(self, batch, batch_idx, dataset_idx=None):
        loss, accuracy = self.evaluate(batch)
        self.log_stage("test", loss, accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.fc.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)

        return [optimizer], [scheduler]

class CelebAModel(BaseModel):
    def __init__(self, lr=1e-4, num_epochs=20, weight=None):
        backbone = torchvision.models.resnet18(weights='DEFAULT')
        fc = nn.Linear(backbone.fc.in_features, 1)
        backbone.fc = nn.Identity()

        super().__init__(backbone, fc, 1, lr, num_epochs, weight)

class CelebAModel2(BaseModel):
    def __init__(self, lr=1e-3, num_epochs=10, weight=None):
        backbone = torchvision.models.vgg16(weights='DEFAULT')
        fc = nn.Linear(backbone.classifier[-1].in_features, 1)
        backbone.classifier[-1] = nn.Identity()

        super().__init__(backbone, fc, 1, lr, num_epochs, weight)

class CIFAR10Model(BaseModel):
    def __init__(self, lr=0.1, momentum=0.9, weight_decay=5e-4, num_epochs=200, weight=None,
            pretrained=False):

        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        backbone = resnet56()
        if pretrained:
            state_dict = torch.load(CIFAR10_MODEL_1_PT, map_location="cpu")['state_dict']
            state_dict["linear.weight"] = state_dict.pop("fc.weight")
            state_dict["linear.bias"] = state_dict.pop("fc.bias")
            backbone.load_state_dict(state_dict)

        fc = backbone.linear
        backbone.linear = nn.Identity()

        super().__init__(backbone, fc, 10, lr, num_epochs, weight)

        self.momentum = momentum
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.fc.parameters(), self.lr, momentum=self.momentum,
            weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 175])

        return [optimizer], [scheduler]

class CIFAR10Model2(BaseModel):
    def __init__(self, lr=1e-3, momentum=0.9, weight_decay=5e-4, num_epochs=20, weight=None):
        backbone = torchvision.models.efficientnet_v2_s(weights='DEFAULT')
        fc = nn.Linear(backbone.classifier[-1].in_features, 10)
        backbone.classifier[-1] = nn.Identity()

        self.momentum = momentum
        self.weight_decay = weight_decay
        super().__init__(backbone, fc, 10, lr, num_epochs, weight)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.fc.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)

        return [optimizer], [scheduler]

class FitzModel(BaseModel):
    def __init__(self, lr=1e-3, num_epochs=30, weight=None):
        backbone = torchvision.models.resnet34(weights='DEFAULT')
        fc = nn.Linear(backbone.fc.in_features, 3)
        backbone.fc = nn.Identity()

        super().__init__(backbone, fc, 3, lr, num_epochs, weight)

class FitzModel2(BaseModel):
    def __init__(self, lr=1e-3, num_epochs=30, weight=None):
        backbone = torchvision.models.efficientnet_v2_m(weights='DEFAULT')
        fc = nn.Linear(backbone.classifier[-1].in_features, 3)
        backbone.classifier[-1] = nn.Identity()

        super().__init__(backbone, fc, 3, lr, num_epochs, weight)
