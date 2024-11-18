import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from lightly.transforms.vicreg_transform import VICRegTransform


class exCIFAR10(Dataset):
    def __init__(self, root, train=True, download=True):
        self.base_transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )

        self.eval_transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                # transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
                # transforms.RandomHorizontalFlip(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )

        # self.train_transform = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        #         transforms.RandomHorizontalFlip(0.5),
        #         transforms.RandomApply(
        #             [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
        #         ),
        #         transforms.RandomGrayscale(0.2),
        #         transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
        #         transforms.RandomSolarize(0.5, p=0.2),
        #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        #     ]
        # )

        self.train_transform = VICRegTransform(
            input_size=32,
            cj_strength=0.5,
            min_scale=0.7,
            gaussian_blur=0.0,
            #    normalize = {"mean": [0.4914, 0.4822, 0.4465], "std": [0.247, 0.243, 0.261]}
        )

        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=None
        )

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        img_original = self.base_transform(img)
        img_train = self.eval_transform(img)
        img_aug1 = self.train_transform(img)
        img_aug2 = self.train_transform(img)
        return img_original, img_train, img_aug1, img_aug2, label

    def __len__(self):
        return len(self.dataset)
