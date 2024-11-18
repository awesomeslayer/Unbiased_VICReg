import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CIFAR10TripleView(Dataset):
    def __init__(self, root, transform, train=True, download=True):
        self.base_transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )
        self.transform = transform
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=None
        )

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img_original = self.base_transform(img)
        img_aug1 = self.transform(img)
        img_aug2 = self.transform(img)
        return img_original, img_aug1, img_aug2, label

    def __len__(self):
        return len(self.dataset)
