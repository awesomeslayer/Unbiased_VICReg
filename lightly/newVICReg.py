import os
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import argparse
from torch import nn
from lightly.loss import VICRegLoss
from lightly.loss.vicreg_loss import VICRegLoss
from lightly.models.modules.heads import VICRegProjectionHead
from lightly.transforms.vicreg_transform import VICRegTransform
from datetime import datetime
import numpy as np
import glob

#todo:
#1)online probing instead linear +- test it
#2)logs on training epochs+steps, loss(and each part), same on evaluation (with lightining) -
#3)tensorborad logs and plots, model and other -

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
        
        # transform for augmentations
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

class VICReg(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = VICRegProjectionHead(
            input_dim=512,
            hidden_dim=512,
            output_dim=512,
            num_layers=3,
        )

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z
    
def online_main(num_epochs = 200, checkpoint_dir="oexp256"):
    batch_size = 256
    lr_linear = 1e-4
    lr_vicreg = 1e-5

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #model VICReg on resnet18 + 512-512-512
    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = VICReg(backbone)
    model.to(device)
    
    linear = nn.Linear(512, 10).to(device)    

    #loss + optimizers
    vicreg_loss = VICRegLoss()

    #edited normilize for CIFAR10
    transform = VICRegTransform(input_size = 32, cj_prob = 0.8, cj_strength = 1.0, cj_bright = 0.8,
                                cj_contrast = 0.8, cj_sat = 0.8, cj_hue = 0.2, min_scale = 0.08, random_gray_scale = 0.2, 
                                gaussian_blur = 0.5, kernel_size = None, sigmas = (0.1, 2), vf_prob = 0.0, hf_prob = 0.5, 
                                rr_prob = 0.0, rr_degrees = None, normalize = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.247, 0.243, 0.261]}) 
    
    # Create datasets for train and test
    train_dataset = CIFAR10TripleView("data/", transform, train=True, download=True)
    test_dataset = CIFAR10TripleView("data/", transform, train=False, download=True)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            drop_last=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           drop_last=False, num_workers=8)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_vicreg)
    linear_optimizer = torch.optim.AdamW(linear.parameters(), lr=lr_linear)
    
    vicreg_start = load_checkpoint(model, optimizer, checkpoint_dir, "vicreg"),
    linear_start = load_checkpoint(linear, linear_optimizer, checkpoint_dir, "linear")
    
    if(vicreg_start[0] == linear_start):
        start_epoch = vicreg_start[0]
    else:
        start_epoch = 0

    if start_epoch < num_epochs:
        print(f"continue from {start_epoch} epoch:")
        model.train()
        linear.train()
        for epoch in range(start_epoch, num_epochs):
            total_loss = 0
            train_loss = 0
            correct = 0
            total = 0
            for batch in train_loader:
                #SSL
                x, x0, x1, y = batch
                x0 = x0[0]
                x1 = x1[0]
                x0, x1 = x0.to(device), x1.to(device)
                z0, z1 = model(x0), model(x1)
                loss = vicreg_loss(z0, z1)
                total_loss += loss.detach()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                #Linear
                x = x.to(device)
                y = y.to(device)
                
                with torch.no_grad(): #dont update backbone weights
                    features = model.backbone(x).flatten(start_dim=1)
                
                linear_optimizer.zero_grad()
                outputs = linear(features)
                loss = nn.CrossEntropyLoss()(outputs, y)
                loss.backward()
                linear_optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

            avg_loss = total_loss / len(train_loader)
            print(f"epoch: {epoch:>02}, VICReg loss: {avg_loss:.5f}")
            
            train_accuracy = 100. * correct / total
            train_loss = train_loss / len(train_loader)
            print(f"epoch: {epoch:>02},"
                    f"Train Loss: {train_loss:.5f}, "
                    f"Train Acc: {train_accuracy:.2f}%")
            
            # Save checkpoints
            save_checkpoint(model, optimizer, epoch, checkpoint_dir, "vicreg")
            save_checkpoint(linear, linear_optimizer, epoch, checkpoint_dir, "linear")
            
        #Testing on each epoch
        model.eval()
        linear.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                x, _, _, y = batch
                x = x.to(device)
                y = y.to(device)
                features = model.backbone(x).flatten(start_dim=1)
                outputs = linear(features)
                loss = nn.CrossEntropyLoss()(outputs, y)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        
        test_accuracy = 100. * correct / total
        test_loss = test_loss / len(test_loader)
        
        print(f"epoch: {epoch:>02},"
                f"Test Loss: {test_loss:.5f}, "
                f"Test Acc: {test_accuracy:.2f}%")

    return model, linear

def linear_main(num_epochs = 100, num_eval_epochs = 30, checkpoint_dir="exp16"):
    lr_vicreg = 1e-4
    lr_linear = 1e-3
    batch_size = 16

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #model VICReg on resnet18 + 512-512-512
    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = VICReg(backbone)
    model.to(device)
    
    linear = nn.Linear(512, 10).to(device)    

    #loss + optimizers
    vicreg_loss = VICRegLoss()

    #edited normilize for CIFAR10
    transform = VICRegTransform(input_size = 32, cj_prob = 0.8, cj_strength = 1.0, cj_bright = 0.8,
                                cj_contrast = 0.8, cj_sat = 0.8, cj_hue = 0.2, min_scale = 0.08, random_gray_scale = 0.2, 
                                gaussian_blur = 0.5, kernel_size = None, sigmas = (0.1, 2), vf_prob = 0.0, hf_prob = 0.5, 
                                rr_prob = 0.0, rr_degrees = None, normalize = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.247, 0.243, 0.261]}) 
    
    # Create datasets for train and test
    train_dataset = CIFAR10TripleView("data/", transform, train=True, download=True)
    test_dataset = CIFAR10TripleView("data/", transform, train=False, download=True)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            drop_last=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           drop_last=False, num_workers=8)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_vicreg)
    linear_optimizer = torch.optim.AdamW(linear.parameters(), lr=lr_linear)
        
    vicreg_start = load_checkpoint(model, None, checkpoint_dir, prefix="vicreg")
    
    if vicreg_start < num_epochs:
        print(f"Continuing VICReg training from epoch {vicreg_start} to {num_epochs}")
        # VICReg Training
        model.train()
        for epoch in range(vicreg_start, num_epochs):
            total_loss = 0
            for batch in train_loader:
                _, x0, x1, _ = batch
                x0 = x0[0]
                x1 = x1[0]
                x0, x1 = x0.to(device), x1.to(device)
                z0, z1 = model(x0), model(x1)
                loss = vicreg_loss(z0, z1)
                total_loss += loss.detach()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            avg_loss = total_loss / len(train_loader)
            print(f"epoch: {epoch:>02}, VICReg loss: {avg_loss:.5f}")
            save_checkpoint(model, optimizer, epoch, checkpoint_dir, prefix="vicreg")
    else:
        print(f"VICReg training already completed on {vicreg_start} epoch")
    
    print(f"linear evaluation epochs to {num_eval_epochs}")
    for epoch in range(0, num_eval_epochs):
        # Training
        model.eval()
        linear.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            x, _, _, y = batch
            y = y.to(device)
            x = x.to(device)
            
            with torch.no_grad():
                features = model.backbone(x).flatten(start_dim=1)
            
            linear_optimizer.zero_grad()
            outputs = linear(features)
            loss = nn.CrossEntropyLoss()(outputs, y)
            loss.backward()
            linear_optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        
        train_accuracy = 100. * correct / total
        train_loss = train_loss / len(train_loader)
        print(f"epoch: {epoch:>02},"
                f"Train Loss: {train_loss:.5f}, "
                f"Train Acc: {train_accuracy:.2f}%")
            
        # Testing
        model.eval()
        linear.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                x, _, _, y = batch
                y = y.to(device)
                x = x.to(device)
                
                features = model.backbone(x).flatten(start_dim=1)
                outputs = linear(features)
                loss = nn.CrossEntropyLoss()(outputs, y)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        
        test_accuracy = 100. * correct / total
        test_loss = test_loss / len(test_loader)
        
        print(f"epoch: {epoch:>02},"
                f"Test Loss: {test_loss:.5f}, "
                f"Test Acc: {test_accuracy:.2f}%")
    
    return model, linear

def setup_logger(checkpoint_dir):
    """Setup logging to both file and console."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = os.path.join(checkpoint_dir, 'log.txt')
    return log_file

def log_message(message, log_file):
    """Log message to both file and console with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(log_file, 'a') as f:
        f.write(log_entry + '\n')

def load_checkpoint(model, optimizer, checkpoint_dir, prefix="vicreg"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'{prefix}_latest.pt')
    
    if not os.path.exists(checkpoint_path):
        return 0
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'] + 1

def save_checkpoint(model, optimizer, epoch, checkpoint_dir, prefix="vicreg"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'{prefix}_latest.pt')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

def parse_args():
    parser = argparse.ArgumentParser(description='VICReg Training with Linear Evaluation')
    parser.add_argument('--batch_size', type=int, default=256,
                      help='batch size for training (default: 256)')
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='number of training epochs (default: 100)')
    parser.add_argument('--num_eval_epochs', type=int, default=100,
                      help='number of evaluation epochs for linear probing (default: 100)')
    parser.add_argument('--checkpoint_dir', type=str, default='exp256/',
                      help='directory to save checkpoints (default: exp256/)')
    parser.add_argument('--lr_vicreg', type=float, default=1e-5,
                      help='learning rate for VICReg training (default: 1e-5)')
    parser.add_argument('--lr_linear', type=float, default=1e-4,
                      help='learning rate for linear evaluation (default: 1e-4)')
    parser.add_argument('--probe', type=str, choices=['linear', 'online'], default='linear',
                      help='probing type: linear or online (default: linear)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    log_file = setup_logger(args.checkpoint_dir)
    
    log_message("Starting training with parameters:", log_file)
    log_message(f"Batch size: {args.batch_size}", log_file)
    log_message(f"Number of epochs: {args.num_epochs}", log_file)
    log_message(f"Number of eval epochs: {args.num_eval_epochs}", log_file)
    log_message(f"Checkpoint directory: {args.checkpoint_dir}", log_file)
    log_message(f"VICReg learning rate: {args.lr_vicreg}", log_file)
    log_message(f"Linear learning rate: {args.lr_linear}", log_file)
    log_message(f"Probing type: {args.probe}", log_file)
    
    if args.probe == 'linear':
        model, linear = linear_main(args, log_file)
    else:  # online
        model, linear = online_main(args, log_file)

if __name__ == "__main__":
    main()

# [Previous online_main and linear_main functions remain unchanged except for using the new logging system]