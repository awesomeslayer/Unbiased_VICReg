import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Helper functions for VICReg loss
def off_diagonal(x):
    # Returns a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def vicreg_loss_biased(x, y, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
    # Invariance loss
    repr_loss = F.mse_loss(x, y)

    # Variance loss
    std_x = torch.sqrt(x.var(dim=0) + 1e-04)
    std_y = torch.sqrt(y.var(dim=0) + 1e-04)
    std_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))

    # Center the embeddings
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)

    # Biased covariance estimation (divide by N)
    N, D = x.size()
    cov_x = (x.T @ x) / N
    cov_y = (y.T @ y) / N

    # Covariance loss
    cov_loss = (off_diagonal(cov_x).pow_(2).sum() / D) + (off_diagonal(cov_y).pow_(2).sum() / D)

    # Total loss
    loss = sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss
    return loss
def vicreg_loss_unbiased(x, y, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
    N, D = x.size()  # N is batch size, D is feature dimension

    # Invariance loss (representation loss)
    repr_loss = F.mse_loss(x, y)
    
    # Concatenate representations
    combined = torch.cat([x, y], dim=0)
    
    # Random permutation of indices and split into two groups
    indices = torch.randperm(N)
    z1_indices = indices[:N//2]  # First half for z1
    z2_indices = indices[N//2:]  # Second half for z2

    z1 = combined[z1_indices]  # List of vectors in z1
    z2 = combined[z2_indices]  # List of vectors in z2

    # Compute covariance matrices
    cov_z1 = sum([z.unsqueeze(1) @ z.unsqueeze(0) for z in z1]) / (N//2 - 1)
    cov_z2 = sum([z.unsqueeze(1) @ z.unsqueeze(0) for z in z2]) / (N//2 - 1)

    # Identity matrix for covariance normalization
    I = torch.eye(D).to(cov_z1.device)

    # Covariance difference <cov_z1 - I, cov_z2 - I>
    cov_diff = (cov_z1 - I) @ (cov_z2 - I)
    
    # Covariance loss based on Frobenius norm
    cov_loss = torch.norm(cov_diff, p='fro')

    # Total loss
    loss = sim_coeff * repr_loss + cov_coeff * cov_loss
    return loss

# Standard VICReg Model
class VICReg(nn.Module):
    def __init__(self, feature_dim=2048):
        super(VICReg, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Output: 32x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Output: 16x16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 8x8
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

# Unbiased VICReg Model (uses the adjusted covariance in the loss function)
class UnbiasedVICReg(VICReg):
    def __init__(self, feature_dim=2048):
        super(UnbiasedVICReg, self).__init__(feature_dim)

# Training setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data augmentation transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load datasets once
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

# Function to create two augmented views
def get_augmented_views(batch):
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert tensor to PIL Image
        transforms.RandomResizedCrop(size=32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),  # Convert back to tensor
    ])
    view1 = torch.stack([transform(img) for img in batch])
    view2 = torch.stack([transform(img) for img in batch])
    return view1.to(device), view2.to(device)


# Linear evaluation
def linear_evaluation(model, device, trainloader, testloader):
    # Freeze the encoder
    for param in model.parameters():
        param.requires_grad = False
    # Define a linear classifier
    classifier = nn.Linear(128, 10).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    # Training loop
    for epoch in range(5):
        classifier.train()
        running_loss = 0.0
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                features = model(data)
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
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            features = model(data)
            outputs = classifier(features)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    accuracy = 100. * correct / total
    return accuracy

# List of batch sizes to experiment with
batch_sizes = [16]
vicreg_accuracies = []
unbiased_vicreg_accuracies = []
num_epochs = 1

for batch_size in batch_sizes:
    print(f"\nRunning experiments with batch size: {batch_size}")
    # Create DataLoaders with the current batch size
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Initialize models
    vicreg = VICReg().to(device)
    unbiased_vicreg = UnbiasedVICReg().to(device)

    # Optimizers
    optimizer_vicreg = optim.Adam(vicreg.parameters(), lr=0.001)
    optimizer_unbiased_vicreg = optim.Adam(unbiased_vicreg.parameters(), lr=0.001)

    # Training biased VICReg
    for epoch in range(num_epochs):  # Reduced epochs for faster execution
        vicreg.train()
        running_loss = 0.0
        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{num_epochs} [VICReg]")
        for (data, _) in progress_bar:
            # Generate two augmented views
            x1, x2 = get_augmented_views(data)
            optimizer_vicreg.zero_grad()
            z1 = vicreg(x1)
            z2 = vicreg(x2)
            loss = vicreg_loss_biased(z1, z2)
            loss.backward()
            optimizer_vicreg.step()
            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        print(f'VICReg Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}')

    # Training UnbiasedVICReg
    for epoch in range(num_epochs):  # Reduced epochs for faster execution
        unbiased_vicreg.train()
        running_loss = 0.0
        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{num_epochs} [UnbiasedVICReg]")
        for (data, _) in progress_bar:
            # Generate two augmented views
            x1, x2 = get_augmented_views(data)
            optimizer_unbiased_vicreg.zero_grad()
            z1 = unbiased_vicreg(x1)
            z2 = unbiased_vicreg(x2)
            loss = vicreg_loss_unbiased(z1, z2)  # Covariance loss is adjusted inside vicreg_loss
            loss.backward()
            optimizer_unbiased_vicreg.step()
            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        print(f'UnbiasedVICReg Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}')

    # Evaluate VICReg
    print("Evaluating VICReg")
    vicreg_accuracy = linear_evaluation(vicreg, device, trainloader, testloader)
    print(f'VICReg Accuracy with batch size {batch_size}: {vicreg_accuracy:.2f}%')
    vicreg_accuracies.append(vicreg_accuracy)

    # Evaluate UnbiasedVICReg
    print("Evaluating UnbiasedVICReg")
    unbiased_vicreg_accuracy = linear_evaluation(unbiased_vicreg, device, trainloader, testloader)
    print(f'Unbiased VICReg Accuracy with batch size {batch_size}: {unbiased_vicreg_accuracy:.2f}%')
    unbiased_vicreg_accuracies.append(unbiased_vicreg_accuracy)

# Plotting the accuracies
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, vicreg_accuracies, marker='o', label='VICReg')
plt.plot(batch_sizes, unbiased_vicreg_accuracies, marker='s', label='Unbiased VICReg')
plt.title('Accuracy vs Batch Size')
plt.xlabel('Batch Size')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig("experiments/vicreg.png")
