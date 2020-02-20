import torch
import torchvision
from torchvision import transforms as transforms

from lib.utils.funcs import auto_change_dir

""" Dataset Loaders"""

"""
from lib.utils.utils import load_samples
from Feature_Json import fake_arg
arg=fake_arg()

"""

def load_mnist(args):
    # Load MNIST
    auto_change_dir("Mnist")

    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),  # ToTensor does min-max normalization.
    ]), )

    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),  # ToTensor does min-max normalization.
    ]), )

    # Create DataLoader
    dataloader_args = dict(shuffle=True, batch_size=args.train_batch_size, num_workers=2)
    train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_data, **dataloader_args)

    return train_loader, test_loader, range(10)

