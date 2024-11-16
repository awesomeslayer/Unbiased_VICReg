import os
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import argparse
from torch import nn
from lightly.loss import VICRegLoss
from lightly.models.modules.heads import VICRegProjectionHead
from lightly.transforms.vicreg_transform import VICRegTransform
from datetime import datetime
import hydra
from omegaconf import DictConfig

class CIFAR10TripleView(Dataset):
    def __init__(self, root, transform, train=True, download=True):
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.247, 0.243, 0.261]
            )
        ])
        self.transform = transform
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, 
            train=train,
            download=download,
            transform=None
        )

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img_original = self.base_transform(img)
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

def online_main(args, log_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_message(f"Using device: {device}", log_file)

    writer = SummaryWriter(log_dir=args.checkpoint_dir)  # Initialize TensorBoard writer

    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = VICReg(backbone)
    model.to(device)
    
    linear = nn.Linear(512, 10).to(device)    
    vicreg_loss = VICRegLoss()

    transform = VICRegTransform(input_size=32)

    train_dataset = CIFAR10TripleView("data/", transform, train=True, download=True)
    test_dataset = CIFAR10TripleView("data/", transform, train=False, download=True)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                             shuffle=True, drop_last=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                            shuffle=False, drop_last=False, num_workers=8)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_vicreg)
    linear_optimizer = torch.optim.AdamW(linear.parameters(), lr=args.lr_linear)
    
    vicreg_start = load_checkpoint(model, optimizer, args.checkpoint_dir, "vicreg")
    linear_start = load_checkpoint(linear, linear_optimizer, args.checkpoint_dir, "linear")

    
    start_epoch = vicreg_start if vicreg_start == linear_start else 0
    
    # Load the first batch
    batch = next(iter(train_loader))
    x, x0, _, y = batch  # x is original, x0 is augmented, y are labels

    # Select 4 samples from the batch for visualization
    num_samples = 4
    x_vis = x[:num_samples]  # Original images
    x0_vis = x0[0][:num_samples]  # First augmentation of the images
    labels_vis = y[:num_samples]  # Corresponding labels

    # Visualize the selected original images
    writer.add_images('Original Images', x_vis, 0)

    # Visualize the selected augmented images
    writer.add_images('Augmented Images', x0_vis, 0)

    # Add labels as text to TensorBoard
    for i in range(num_samples):
        writer.add_text(f'Label_{i}', f'Label: {labels_vis[i].item()}', 0)

    # Visualize the model graph (using the first batch)
    writer.add_graph(model, x_vis.to(device))

    if start_epoch < args.num_epochs:
        log_message(f"Continuing training from epoch {start_epoch} to {args.num_epochs}", log_file)
        model.train()
        linear.train()
        
        for epoch in range(start_epoch, args.num_epochs):
            total_loss = 0
            train_loss = 0
            correct = 0
            total = 0

            for batch_idx, batch in enumerate(train_loader):
                # SSL training
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

                # Linear classifier training
                x = x.to(device)
                y = y.to(device)
                
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

            avg_loss = total_loss / len(train_loader)
            train_accuracy = 100. * correct / total
            train_loss = train_loss / len(train_loader)

            # Log train loss and accuracy to TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('VICReg Loss/train', avg_loss, epoch)

            log_message(f"Epoch: {epoch:>02}, VICReg loss: {avg_loss:.5f}, "
                       f"Train Loss: {train_loss:.5f}, Train Acc: {train_accuracy:.2f}%", log_file)

            # Visualize the first batch
            if epoch == 0 and batch_idx == 0:
                img_grid = torchvision.utils.make_grid(x)
                writer.add_image('First Batch/train', img_grid)

            save_checkpoint(model, optimizer, epoch, args.checkpoint_dir, "vicreg")
            save_checkpoint(linear, linear_optimizer, epoch, args.checkpoint_dir, "linear")

            # Testing
            model.eval()
            linear.eval()
            test_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    x, _, _, y = batch
                    x, y = x.to(device), y.to(device)
                    features = model.backbone(x).flatten(start_dim=1)
                    outputs = linear(features)
                    loss = nn.CrossEntropyLoss()(outputs, y)
                    
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += y.size(0)
                    correct += predicted.eq(y).sum().item()
            
            test_accuracy = 100. * correct / total
            test_loss = test_loss / len(test_loader)

            # Log test loss and accuracy to TensorBoard
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/test', test_accuracy, epoch)

            log_message(f"Epoch: {epoch:>02}, Test Loss: {test_loss:.5f}, Test Acc: {test_accuracy:.2f}%", log_file)

    else:
        log_message(f"Training already completed on {start_epoch} epoch", log_file)

    writer.close()  # Close TensorBoard writer at the end

    return model, linear

def linear_main(args, log_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_message(f"Using device: {device}", log_file)

    # Initialize the TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, "tensorboard_logs"))

    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = VICReg(backbone)
    model.to(device)
    
    linear = nn.Linear(512, 10).to(device)

    vicreg_loss = VICRegLoss()

    transform = VICRegTransform(input_size=32, cj_prob=0.8, cj_strength=1.0, cj_bright=0.8,
                                cj_contrast=0.8, cj_sat=0.8, cj_hue=0.2, min_scale=0.08,
                                random_gray_scale=0.2, gaussian_blur=0.5, kernel_size=None,
                                sigmas=(0.1, 2), vf_prob=0.0, hf_prob=0.5, rr_prob=0.0,
                                rr_degrees=None, normalize={'mean': [0.4914, 0.4822, 0.4465],
                                                            'std': [0.247, 0.243, 0.261]})

    train_dataset = CIFAR10TripleView("data/", transform, train=True, download=True)
    test_dataset = CIFAR10TripleView("data/", transform, train=False, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, drop_last=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False, drop_last=False, num_workers=8)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_vicreg)
    linear_optimizer = torch.optim.AdamW(linear.parameters(), lr=args.lr_linear)

    # Load checkpoints if available
    vicreg_start = load_checkpoint(model, optimizer, args.checkpoint_dir, prefix="vicreg")
    start_epoch = vicreg_start

    # Load the first batch
    batch = next(iter(train_loader))
    x, x0, _, y = batch  # x is original, x0 is augmented, y are labels

    # Select 4 samples from the batch for visualization
    num_samples = 4
    x_vis = x[:num_samples]  # Original images
    x0_vis = x0[0][:num_samples]  # First augmentation of the images
    labels_vis = y[:num_samples]  # Corresponding labels

    # Visualize the selected original images
    writer.add_images('Original Images', x_vis, 0)

    # Visualize the selected augmented images
    writer.add_images('Augmented Images', x0_vis, 0)

    # Add labels as text to TensorBoard
    for i in range(num_samples):
        writer.add_text(f'Label_{i}', f'Label: {labels_vis[i].item()}', 0)

    # Visualize the model graph (using the first batch)
    writer.add_graph(model, x_vis.to(device))

    if start_epoch < args.num_epochs:
        log_message(f"Continuing VICReg training from epoch {vicreg_start} to {args.num_epochs}", log_file)
        model.train()
        for epoch in range(vicreg_start, args.num_epochs):
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
            log_message(f"Epoch: {epoch:>02}, VICReg loss: {avg_loss:.5f}", log_file)

            # TensorBoard logging for training loss
            writer.add_scalar('VICReg_loss/train', avg_loss.item(), epoch)

            save_checkpoint(model, optimizer, epoch, args.checkpoint_dir, prefix="vicreg")
    else:
        log_message(f"VICReg training already completed on {vicreg_start} epoch", log_file)

    log_message(f"Starting linear evaluation for {args.num_eval_epochs} epochs", log_file)
    for epoch in range(args.num_eval_epochs):
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
        log_message(f"Epoch: {epoch:>02}, Train Loss: {train_loss:.5f}, Train Acc: {train_accuracy:.2f}%", log_file)

        # TensorBoard logging for train loss and accuracy
        writer.add_scalar('Train_loss', train_loss, epoch)
        writer.add_scalar('Train_accuracy', train_accuracy, epoch)

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
        log_message(f"Epoch: {epoch:>02}, Test Loss: {test_loss:.5f}, Test Acc: {test_accuracy:.2f}%", log_file)

        # TensorBoard logging for test loss and accuracy
        writer.add_scalar('Test_loss', test_loss, epoch)
        writer.add_scalar('Test_accuracy', test_accuracy, epoch)

    # Close the TensorBoard writer
    writer.close()

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
    parser.add_argument('--checkpoint_dir', type=str, default='linear/256/',
                      help='directory to save checkpoints (default: exp256/)')
    parser.add_argument('--lr_vicreg', type=float, default=1e-5,
                      help='learning rate for VICReg training (default: 1e-5)')
    parser.add_argument('--lr_linear', type=float, default=1e-4,
                      help='learning rate for linear evaluation (default: 1e-4)')
    parser.add_argument('--probe', type=str, choices=['linear', 'online'], default='linear',
                      help='probing type: linear or online (default: linear)')
    
    return parser.parse_args()
@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    log_file = setup_logger(cfg.checkpoint_dir)

    log_message("Starting training with parameters:", log_file)
    log_message(f"Batch size: {cfg.batch_size}", log_file)
    log_message(f"Number of epochs: {cfg.num_epochs}", log_file)
    log_message(f"Number of eval epochs: {cfg.num_eval_epochs}", log_file)
    log_message(f"Checkpoint directory: {cfg.checkpoint_dir}", log_file)
    log_message(f"VICReg learning rate: {cfg.lr_vicreg}", log_file)
    log_message(f"Linear learning rate: {cfg.lr_linear}", log_file)
    log_message(f"Probing type: {cfg.probe}", log_file)

    if cfg.probe == 'linear':
        model, linear = linear_main(cfg, log_file)
    else:  # online
        model, linear = online_main(cfg, log_file)

if __name__ == "__main__":
    main()