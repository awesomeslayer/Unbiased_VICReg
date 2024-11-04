import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data augmentation transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load datasets
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

def get_augmented_views(batch):
    # Convert tensors to PIL Images
    pil_images = [transforms.ToPILImage()(img) for img in batch]
    xi = torch.stack([train_transform(img) for img in pil_images])
    xj = torch.stack([train_transform(img) for img in pil_images])
    return xi.to(device), xj.to(device)

# Helper functions for VICReg loss (same as in the VICReg code)
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def vicreg_loss_biased(x, y, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
    repr_loss = nn.functional.mse_loss(x, y)
    std_x = torch.sqrt(x.var(dim=0) + 1e-04)
    std_y = torch.sqrt(y.var(dim=0) + 1e-04)
    std_loss = torch.mean(nn.functional.relu(1 - std_x)) + torch.mean(nn.functional.relu(1 - std_y))
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    N, D = x.size()
    cov_x = (x.T @ x) / N
    cov_y = (y.T @ y) / N
    cov_loss = (off_diagonal(cov_x).pow_(2).sum() / D) + (off_diagonal(cov_y).pow_(2).sum() / D)
    loss = sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss
    return loss

def vicreg_loss_unbiased(x, y, sim_coeff=25.0, cov_coeff=1.0):
    N, D = x.size()
    repr_loss = nn.functional.mse_loss(x, y)
    combined = torch.cat([x, y], dim=0)
    indices = torch.randperm(N)
    z1 = combined[indices[:N//2]]
    z2 = combined[indices[N//2:]]
    cov_z1 = sum([z.unsqueeze(1) @ z.unsqueeze(0) for z in z1]) / (N//2 - 1)
    cov_z2 = sum([z.unsqueeze(1) @ z.unsqueeze(0) for z in z2]) / (N//2 - 1)
    I = torch.eye(D).to(cov_z1.device)
    cov_diff = (cov_z1 - I) @ (cov_z2 - I)
    cov_loss = torch.norm(cov_diff, p='fro')
    loss = sim_coeff * repr_loss + cov_coeff * cov_loss
    return loss

# VICReg and SimCLR models
class VICReg(nn.Module):
    def __init__(self, feature_dim=2048):
        super(VICReg, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, feature_dim),
            nn.ReLU(),
        )
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        z = self.encoder(x)
        h = self.projector(z)
        return h

class UnbiasedVICReg(VICReg):
    def __init__(self, feature_dim=2048):
        super(UnbiasedVICReg, self).__init__(feature_dim)

class SimCLR(nn.Module):
    def __init__(self, feature_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.projector = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z

# NT-Xent Loss for SimCLR
class NT_XentLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NT_XentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.mask = self._get_correlated_mask().type(torch.bool)

    def _get_correlated_mask(self):
        N = 2 * self.batch_size
        mask = torch.ones((N, N)) - torch.eye(N)
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positives = torch.cat([sim_i_j, sim_j_i], dim=0)
        negatives = sim[self.mask].view(N, -1)
        labels = torch.zeros(N).to(device).long()
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

# Training and evaluation loop (shared for all models)
def linear_evaluation(encoder, device, train_loader, test_loader, feature_dim):
    for param in encoder.parameters():
        param.requires_grad = False
    classifier = nn.Linear(feature_dim, 10).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.03)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(5):
        classifier.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                features = encoder(data)
            optimizer.zero_grad()
            output = classifier(features)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            features = encoder(data)
            outputs = classifier(features)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    accuracy = 100. * correct / total
    return accuracy

# Experiment with different batch sizes and plot accuracies
#batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
batch_size = [4]
num_epochs = 1
vicreg_accuracies = []
unbiased_vicreg_accuracies = []
simclr_accuracies = []

# Function to plot and save loss curves
def plot_loss_curve(losses, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label=f'{model_name} Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"experiments/loss/{model_name}.png")

for batch_size in batch_sizes:
    print(f"\nRunning experiments with batch size: {batch_size}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    
    # VICReg
    vicreg_model = VICReg().to(device)
    optimizer_vicreg = optim.Adam(vicreg_model.parameters(), lr=0.03)
    vicreg_losses = []
    for epoch in range(num_epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [SimCLR]")

        vicreg_model.train()
        epoch_loss = 0
        for (data, _) in progress_bar:
            x1, x2 = get_augmented_views(data)
            optimizer_vicreg.zero_grad()
            z1 = vicreg_model(x1)
            z2 = vicreg_model(x2)
            loss = vicreg_loss_biased(z1, z2)
            loss.backward()
            optimizer_vicreg.step()
            epoch_loss += loss.item()
        vicreg_losses.append(epoch_loss / len(train_loader))

    plot_loss_curve(vicreg_losses, f'VICReg_{batch_size}')
    vicreg_acc = linear_evaluation(vicreg_model.encoder, device, train_loader, test_loader, 2048)
    vicreg_accuracies.append(vicreg_acc)

    # Unbiased VICReg
    unbiased_vicreg_model = UnbiasedVICReg().to(device)
    optimizer_unbiased_vicreg = optim.Adam(unbiased_vicreg_model.parameters(), lr=0.03)
    unbiased_vicreg_losses = []
    for epoch in range(num_epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [SimCLR]")

        unbiased_vicreg_model.train()
        epoch_loss = 0
        for (data, _) in progress_bar:
            x1, x2 = get_augmented_views(data)
            optimizer_unbiased_vicreg.zero_grad()
            z1 = unbiased_vicreg_model(x1)
            z2 = unbiased_vicreg_model(x2)
            loss = vicreg_loss_unbiased(z1, z2)
            loss.backward()
            optimizer_unbiased_vicreg.step()
            epoch_loss += loss.item()
        unbiased_vicreg_losses.append(epoch_loss / len(train_loader))
    plot_loss_curve(unbiased_vicreg_losses, f'UnbiasedVICReg_{batch_size}')
    unbiased_vicreg_acc = linear_evaluation(unbiased_vicreg_model.encoder, device, train_loader, test_loader, 2048)
    unbiased_vicreg_accuracies.append(unbiased_vicreg_acc)

    # SimCLR
    simclr_model = SimCLR().to(device)
    optimizer_simclr = optim.Adam(simclr_model.parameters(), lr=0.03)
    criterion_simclr = NT_XentLoss(batch_size=batch_size, temperature=0.5)
    simclr_losses = []
    for epoch in range(num_epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [SimCLR]")

        simclr_model.train()
        epoch_loss = 0
        for (data, _) in progress_bar:
            x1, x2 = get_augmented_views(data)
            optimizer_simclr.zero_grad()
            _, z1 = simclr_model(x1)
            _, z2 = simclr_model(x2)
            loss = criterion_simclr(z1, z2)
            loss.backward()
            optimizer_simclr.step()
            epoch_loss += loss.item()
        simclr_losses.append(epoch_loss / len(train_loader))
    plot_loss_curve(simclr_losses, f'SimCLR_{batch_size}')
    simclr_acc = linear_evaluation(simclr_model.encoder, device, train_loader, test_loader, 256)
    simclr_accuracies.append(simclr_acc)

# Plot accuracies
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, vicreg_accuracies, label='VICReg', marker='o')
plt.plot(batch_sizes, unbiased_vicreg_accuracies, label='UnbiasedVICReg', marker='o')
plt.plot(batch_sizes, simclr_accuracies, label='SimCLR', marker='o')
plt.xlabel('Batch Size')
plt.ylabel('Accuracy')
plt.title(f'Accuracy vs. Batch Size with n_epochs = {num_epochs} for VICReg, UnbiasedVICReg, and SimCLR')
plt.legend()
plt.grid(True)
plt.savefig(f"experiments/main_plot_{num_epochs}.png")
