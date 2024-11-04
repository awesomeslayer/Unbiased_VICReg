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

# SimCLR Model Definition
class SimCLR(nn.Module):
    def __init__(self, feature_dim=128):
        super(SimCLR, self).__init__()
        # Encoder Network (Simple CNN)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Output: 32x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Output: 16x16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 8x8
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        # Projection Head
        self.projector = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z

# NT-Xent Loss Function
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

# Data Augmentation
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

# Load Datasets
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

# Function to create two augmented views
def get_augmented_views(batch):
    # Convert tensors to PIL Images
    pil_images = [transforms.ToPILImage()(img) for img in batch]
    xi = torch.stack([train_transform(img) for img in pil_images])
    xj = torch.stack([train_transform(img) for img in pil_images])
    return xi.to(device), xj.to(device)


# Linear Evaluation
def linear_evaluation(encoder, device, train_loader, test_loader):
    # Freeze the encoder
    for param in encoder.parameters():
        param.requires_grad = False
    # Define a linear classifier
    classifier = nn.Linear(256, 10).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    # Training loop
    for epoch in range(5):
        classifier.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                features = encoder(data)
            optimizer.zero_grad()
            output = classifier(features)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    # Testing
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

# List of batch sizes to experiment with
batch_sizes = [16]
simclr_accuracies = []
num_epochs = 1
for batch_size in batch_sizes:
    print(f"\nRunning experiments with batch size: {batch_size}")
    # Create DataLoaders with the current batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    simclr_model = SimCLR().to(device)
    optimizer = optim.Adam(simclr_model.parameters(), lr=0.001)
    nt_xent_criterion = NT_XentLoss(batch_size, temperature=0.5)

    # Training SimCLR
    for epoch in range(num_epochs):  # Adjust number of epochs as needed
        simclr_model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [SimCLR]")
        for (data, _) in progress_bar:
            xi, xj = get_augmented_views(data)
            optimizer.zero_grad()
            _, zi = simclr_model(xi)
            _, zj = simclr_model(xj)
            loss = nt_xent_criterion(zi, zj)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        print(f'SimCLR Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}')

    # Evaluate SimCLR
    print("Evaluating SimCLR")
    # Pass the encoder directly, not a function
    simclr_accuracy = linear_evaluation(simclr_model.encoder, device, train_loader, test_loader)
    print(f'SimCLR Accuracy with batch size {batch_size}: {simclr_accuracy:.2f}%')
    simclr_accuracies.append(simclr_accuracy)

    # Save the model weights
    os.makedirs('weights', exist_ok=True)
    torch.save(simclr_model.state_dict(), f'weights/simclr_weights_batchsize_{batch_size}.pth')

# Plotting the accuracies
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, simclr_accuracies, marker='o', label='SimCLR')
plt.title('Accuracy vs Batch Size for SimCLR')
plt.xlabel('Batch Size')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig("experiments/simclr.png")
