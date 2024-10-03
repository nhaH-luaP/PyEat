import lightning as L
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, root, batch_size):
        super().__init__()
        self.mean = 0
        self.std = 1
        self.root = root
        self.batch_size = batch_size

    def train_dataloader(self):
        # Shape is (1, 28, 28) so we pad it to fit the Spectogramm Dimensions (1024, 128)
        transform = T.Compose([T.Pad(padding=((1024-28)//2, (128-28)//2)), T.ToTensor(), T.Normalize(self.mean, self.std)])
        dataset = MNIST(
            root=self.root,
            train=True,
            transform=transform,
            download=True,
        )
        dataloader = DataLoader(
            Subset(dataset, range(6)),
            batch_size=self.batch_size,
            num_workers=7,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True
        )
        return dataloader
    
    def val_dataloader(self):
        transform = T.Compose([T.Pad(padding=((1024-28)//2, (128-28)//2)), T.ToTensor(), T.Normalize(self.mean, self.std)])
        dataset = MNIST(
            root=self.root,
            train=False,
            transform=transform,
            download=True,
        )
        dataloader = DataLoader(
            Subset(dataset, range(6)),
            batch_size=self.batch_size,
            num_workers=7,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True
        )
        return dataloader
    
    def test_dataloader(self):
        return self.val_dataloader()