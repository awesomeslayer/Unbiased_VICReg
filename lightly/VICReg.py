import os
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch import nn
from lightly.loss import VICRegLoss
from lightly.loss.vicreg_loss import VICRegLoss
from lightly.models.modules.heads import VICRegProjectionHead
from lightly.transforms.vicreg_transform import VICRegTransform
import numpy as np
import glob

#todo:
#0)good augumentations for cifar-10 +
#1)online probing instead linear +-
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

def load_checkpoint(model, optimizer, checkpoint_dir, prefix="vicreg"):
    """
    Load the latest checkpoint for the specified model type.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into (can be None)
        checkpoint_dir: Directory containing checkpoints
        prefix: 'vicreg' or 'linear' to specify which type of checkpoint to load
    
    Returns:
        start_epoch: Next epoch to start training from
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'{prefix}_latest.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"No {prefix} checkpoint found in {checkpoint_dir}")
        return 0
    
    print(f"Loading {prefix} checkpoint")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'] + 1

def save_checkpoint(model, optimizer, epoch, checkpoint_dir, prefix="vicreg"):
    """Save the latest checkpoint, overwriting the previous one."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'{prefix}_latest.pt')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

def online_main(num_epochs = 10, checkpoint_dir="exp256"):
    #device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #model VICReg on resnet18 + 512-512-512
    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = VICReg(backbone)
    model.to(device)
    
    linear = nn.Linear(512, 10).to(device)    

    #loss + optimizers
    vicreg_loss = VICRegLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    linear_optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

    #edited normilize for CIFAR10
    transform = VICRegTransform(input_size = 32, cj_prob = 0.8, cj_strength = 1.0, cj_bright = 0.8,
                                cj_contrast = 0.8, cj_sat = 0.8, cj_hue = 0.2, min_scale = 0.08, random_gray_scale = 0.2, 
                                gaussian_blur = 0.5, kernel_size = None, sigmas = (0.1, 2), vf_prob = 0.0, hf_prob = 0.5, 
                                rr_prob = 0.0, rr_degrees = None, normalize = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.247, 0.243, 0.261]}) 
    
    # Create datasets for train and test
    train_dataset = CIFAR10TripleView("data/", transform, train=True, download=True)
    test_dataset = CIFAR10TripleView("data/", transform, train=False, download=True)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, 
                            drop_last=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, 
                           drop_last=False, num_workers=8)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    linear_optimizer = torch.optim.AdamW(linear.parameters(), lr=1e-3)
    
    vicreg_start = load_checkpoint(model, optimizer, checkpoint_dir, "vicreg"),
    linear_start = load_checkpoint(linear, linear_optimizer, checkpoint_dir, "linear")
    
    if(vicreg_start == linear_start):
        start_epoch = vicreg_start
    else:
        start_epoch = 0

    if start_epoch < num_epochs:
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

def linear_main(num_epochs = 10, num_eval_epochs = 5, checkpoint_dir="exp256", probing='linear'):
    #device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #model VICReg on resnet18 + 512-512-512
    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = VICReg(backbone)
    model.to(device)
    
    linear = nn.Linear(512, 10).to(device)    

    #loss + optimizers
    vicreg_loss = VICRegLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    linear_optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

    #edited normilize for CIFAR10
    transform = VICRegTransform(input_size = 32, cj_prob = 0.8, cj_strength = 1.0, cj_bright = 0.8,
                                cj_contrast = 0.8, cj_sat = 0.8, cj_hue = 0.2, min_scale = 0.08, random_gray_scale = 0.2, 
                                gaussian_blur = 0.5, kernel_size = None, sigmas = (0.1, 2), vf_prob = 0.0, hf_prob = 0.5, 
                                rr_prob = 0.0, rr_degrees = None, normalize = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.247, 0.243, 0.261]}) 
    
    # Create datasets for train and test
    train_dataset = CIFAR10TripleView("data/", transform, train=True, download=True)
    test_dataset = CIFAR10TripleView("data/", transform, train=False, download=True)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, 
                            drop_last=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, 
                           drop_last=False, num_workers=8)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    linear_optimizer = torch.optim.AdamW(linear.parameters(), lr=1e-3)
        
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
    
    linear_start = load_checkpoint(linear, linear_optimizer, checkpoint_dir, prefix="linear")
    
    if linear_start < num_eval_epochs:
        print(f"Continuing linear evaluation from epoch {linear_start} to {num_eval_epochs}")
        for epoch in range(linear_start, num_eval_epochs):
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
            save_checkpoint(linear, linear_optimizer, epoch, checkpoint_dir, prefix="linear")
            
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
    else:
        print(f"Linear training already completed on {linear_start} epoch")
        print(f"Testing on this epoch")
        model.eval()
        linear.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                x, _, _, y = batch
                x = x[0]
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


if __name__ == '__main__':
    #online_main()
    linear_main()
