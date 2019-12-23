import torch
from torchvision import datasets, transforms

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor



def load_dataset(args,kwargs):
    transformc = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # transforms.Lambda(random_return),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transformc)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transformc)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    return test_loader, train_loader
