# probing.py

import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import lightly
from lightly.loss import VICRegLoss
from lightly.models.modules.heads import VICRegProjectionHead
from lightly.transforms.vicreg_transform import VICRegTransform
from pytorch_lightning.loggers import TensorBoardLogger

def off_diagonal(x):
    """Return the off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class VICRegLossWithComponents(VICRegLoss):
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, eps=1e-4, gamma=1.0):
        super().__init__()  # Initialize the base class without extra arguments
        # Set coefficients manually
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.eps = eps
        self.gamma = gamma

    def forward(self, x0: torch.Tensor, x1: torch.Tensor):
        # Invariance Loss (Mean Squared Error between representations)
        repr_loss = F.mse_loss(x0, x1)

        # Compute Variance and Covariance
        x0 = x0 - x0.mean(dim=0)
        x1 = x1 - x1.mean(dim=0)

        # Compute covariance loss
        N, D = x0.size()
        cov_x0 = (x0.T @ x0) / (N - 1)
        cov_x1 = (x1.T @ x1) / (N - 1)
        cov_loss = (off_diagonal(cov_x0).pow(2).sum()) / D + (off_diagonal(cov_x1).pow(2).sum()) / D

        # Total VICReg loss
        loss = (self.sim_coeff * repr_loss) + (self.cov_coeff * cov_loss)

        return loss, repr_loss, cov_loss

class CIFAR10TripleView(Dataset):
    def __init__(self, root, transform, train=True, download=True):
        # Basic transform for original image (just normalization)
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.247, 0.243, 0.261]
            )
        ])

        # Transform for augmentations
        self.transform = transform

        # Load the original dataset
        self.dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=None  # No transform here as we'll apply them manually
        )

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        # Get original image with only normalization
        img_original = self.base_transform(img)

        # Get two augmented versions using VICReg transform
        img_aug1, img_aug2 = self.transform(img)

        return img_original, img_aug1, img_aug2, label

    def __len__(self):
        return len(self.dataset)

class SemiSSL(pl.LightningModule):
    def __init__(self, num_classes=10, lr_vicreg=1e-4, lr_linear=1e-3, weight_decay=1e-6):
        super().__init__()
        # Model setup
        resnet = torchvision.models.resnet18()
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.projection_head = VICRegProjectionHead(
            input_dim=512,
            hidden_dim=512,
            output_dim=512,
            num_layers=3,
        )
        self.linear = nn.Linear(512, num_classes)
        self.vicreg_loss_fn = VICRegLossWithComponents()
        self.lr_vicreg = lr_vicreg
        self.lr_linear = lr_linear
        self.weight_decay = weight_decay

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        return features

    def shared_step(self, batch):
        x_original, x_aug1, x_aug2, y = batch

        # VICReg projections for augmentations
        z1 = self.projection_head(self.forward(x_aug1))
        z2 = self.projection_head(self.forward(x_aug2))

        # VICReg loss
        vicreg_loss, invariance_loss, covariance_loss = self.vicreg_loss_fn(z1, z2)

        # Classification logits
        features = self.forward(x_original)
        logits = self.linear(features)
        classification_loss = F.cross_entropy(logits, y)

        # Metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        return {
            'vicreg_loss': vicreg_loss,
            'classification_loss': classification_loss,
            'invariance_loss': invariance_loss,
            'covariance_loss': covariance_loss,
            'logits': logits,
            'y': y,
            'accuracy': acc
        }

    def training_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        loss = outputs['vicreg_loss'] + outputs['classification_loss']

        # Logging
        self.log('train/vicreg_loss', outputs['vicreg_loss'], on_step=False, on_epoch=True)
        self.log('train/classification_loss', outputs['classification_loss'], on_step=False, on_epoch=True)
        self.log('train/loss', loss, on_step=False, on_epoch=True)
        self.log('train/invariance_loss', outputs['invariance_loss'], on_step=False, on_epoch=True)
        self.log('train/covariance_loss', outputs['covariance_loss'], on_step=False, on_epoch=True)
        self.log('train/accuracy', outputs['accuracy'], on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        loss = outputs['classification_loss']

        # Logging
        self.log('val/classification_loss', loss, on_step=False, on_epoch=True)
        self.log('val/accuracy', outputs['accuracy'], on_step=False, on_epoch=True)
        self.log('val/vicreg_loss', outputs['vicreg_loss'], on_step=False, on_epoch=True)
        self.log('val/invariance_loss', outputs['invariance_loss'], on_step=False, on_epoch=True)
        self.log('val/covariance_loss', outputs['covariance_loss'], on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {'params': self.backbone.parameters(), 'lr': self.lr_vicreg},
            {'params': self.projection_head.parameters(), 'lr': self.lr_vicreg},
            {'params': self.linear.parameters(), 'lr': self.lr_linear}
        ], weight_decay=self.weight_decay)
        return optimizer

    def prepare_data(self):
        # Data transformations
        self.transform = VICRegTransform(
            input_size=32,
            cj_prob=0.8,
            cj_strength=1.0,
            cj_bright=0.8,
            cj_contrast=0.8,
            cj_sat=0.8,
            cj_hue=0.2,
            min_scale=0.08,
            random_gray_scale=0.2,
            gaussian_blur=0.5,
            kernel_size=None,
            sigmas=(0.1, 2),
            vf_prob=0.0,
            hf_prob=0.5,
            rr_prob=0.0,
            rr_degrees=None,
            normalize={'mean': [0.4914, 0.4822, 0.4465],
                       'std': [0.247, 0.243, 0.261]}
        )

        # Datasets
        self.train_dataset = CIFAR10TripleView("data/", self.transform, train=True, download=True)
        self.val_dataset = CIFAR10TripleView("data/", self.transform, train=False, download=True)

    def setup(self, stage=None):
        # Data loaders
        batch_size = 256
        num_workers = 8

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

def semi_ssl(num_epochs=10, checkpoint_dir="lightning_logs"):
    # Logger and Trainer setup
    logger = TensorBoardLogger(save_dir=checkpoint_dir, name='vicreg_semi_ssl')
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        logger=logger,
        log_every_n_steps=50,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor='val/classification_loss')]
    )

    # Model instantiation
    model = SemiSSL(
        num_classes=10,
        lr_vicreg=1e-4,
        lr_linear=1e-3,
        weight_decay=1e-6
    )

    # Training
    trainer.fit(model)

if __name__ == '__main__':
    semi_ssl()
