from torchvision.datasets import MNIST
import lightning as L
from torch.utils.data import DataLoader
import torchvision.transforms as T


class LitMNISTDataModule(L.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.mean = 0.1307
        self.std = 0.3081
        self.batch_size = batch_size
        
    def train_dataloader(self):
        transform = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std), T.Pad(padding=(2, 2))]) # , T.Pad(padding=((128-28)//2, (1024-28)//2))])
        dataset = MNIST(
            root='/home/phahn/datasets/MNIST/',
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
        transform = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std), T.Pad(padding=(2, 2))]) # , T.Pad(padding=((128-28)//2, (1024-28)//2))])
        dataset = MNIST(
            root='/home/phahn/datasets/MNIST',
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