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
        img_aug1 = self.transform(img)
        img_aug2 = self.transform(img)

        return img_original, img_aug1, img_aug2, label

    def __len__(self):
        return len(self.dataset)


class VICRegModel(pl.LightningModule):
    def __init__(self, backbone, num_classes, lr_vicreg, lr_linear, weight_decay):
        super().__init__()
        self.backbone = backbone
        self.projection_head = VICRegProjectionHead(
            input_dim=512,
            hidden_dim=512,
            output_dim=512,
            num_layers=3,
        )
        self.linear = nn.Linear(512, num_classes)
        self.vicreg_loss_fn = VICRegLoss()
        self.lr_vicreg = lr_vicreg
        self.lr_linear = lr_linear
        self.weight_decay = weight_decay

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        return features

    def shared_step(self, batch):
        x_original, x_aug1, x_aug2, y = batch
        x_aug1 = x_aug1[0]
        x_aug2 = x_aug2[0]

        x_original = x_original.to(self.device)
        x_aug1 = x_aug1.to(self.device)
        x_aug2 = x_aug2.to(self.device)
        y = y.to(self.device)

        # VICReg projections
        z1 = self.projection_head(self.forward(x_aug1))
        z2 = self.projection_head(self.forward(x_aug2))

        # VICReg loss
        vicreg_loss = self.vicreg_loss_fn(z1, z2)

        # Classification logits
        with torch.no_grad():
            features = self.forward(x_original).detach()
        logits = self.linear(features)
        classification_loss = F.cross_entropy(logits, y)

        return {
            'vicreg_loss': vicreg_loss,
            'classification_loss': classification_loss,
            'logits': logits,
            'y': y
        }


    def training_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        loss = outputs['vicreg_loss'] + outputs['classification_loss']

        # Logging
        self.log('train/vicreg_loss', outputs['vicreg_loss'], on_step=False, on_epoch=True)
        self.log('train/classification_loss', outputs['classification_loss'], on_step=False, on_epoch=True)
        self.log('train/loss', loss, on_step=False, on_epoch=True)

        # Metrics
        preds = torch.argmax(outputs['logits'], dim=1)
        acc = (preds == outputs['y']).float().mean()
        self.log('train/accuracy', acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        loss = outputs['vicreg_loss'] + outputs['classification_loss']

        # Logging
        self.log('val/vicreg_loss', outputs['vicreg_loss'], on_step=False, on_epoch=True)
        self.log('val/classification_loss', outputs['classification_loss'], on_step=False, on_epoch=True)
        self.log('val/loss', loss, on_step=False, on_epoch=True)

        # Metrics
        preds = torch.argmax(outputs['logits'], dim=1)
        acc = (preds == outputs['y']).float().mean()
        self.log('val/accuracy', acc, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {'params': list(self.backbone.parameters()) + list(self.projection_head.parameters()), 'lr': self.lr_vicreg},
            {'params': self.linear.parameters(), 'lr': self.lr_linear}
        ], weight_decay=self.weight_decay)
        return optimizer


def online_main(num_epochs=10, checkpoint_dir="lightning_logs"):
    batch_size = 256
    lr_linear = 1e-3
    lr_vicreg = 1e-4
    weight_decay = 1e-6

    # Model setup
    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])

    # Data transformations
    transform = VICRegTransform(
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

    # Datasets and loaders
    train_dataset = CIFAR10TripleView("data/", transform, train=True, download=True)
    val_dataset = CIFAR10TripleView("data/", transform, train=False, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)

    # Model instantiation
    model = VICRegModel(
        backbone=backbone,
        num_classes=10,
        lr_vicreg=lr_vicreg,
        lr_linear=lr_linear,
        weight_decay=weight_decay
    )

    # Logger and Trainer
    logger = TensorBoardLogger(save_dir=checkpoint_dir, name='vicreg_online')
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        logger=logger,
        log_every_n_steps=50,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor='val/loss')]
    )

    # Training
    trainer.fit(model, train_loader, val_loader)

    return model


if __name__ == '__main__':
    online_main()
