
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms

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
