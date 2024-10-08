from torchvision.datasets import CIFAR10
import lightning as L
from torch.utils.data import DataLoader
import torchvision.transforms as T


class LitCIFAR10DataModule(L.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.247, 0.243, 0.262)
        self.batch_size = batch_size
        
    def train_dataloader(self):
        transform = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(self.mean, self.std)])
        dataset = CIFAR10(
            root='/home/phahn/datasets/CIFAR/',
            train=True,
            transform=transform,
            download=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader
    
    def val_dataloader(self):
        transform = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        dataset = CIFAR10(
            root='/home/phahn/datasets/CIFAR/',
            train=False,
            transform=transform,
            download=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=1,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader
    
    def test_dataloader(self):
        return self.val_dataloader()