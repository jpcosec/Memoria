import torch
from torch.nn import functional as F
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


def save_samples(tensor, start=0, folder="samples"):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    from numpy import split, squeeze
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    #print(tensor.shape)
    ndarr = tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(0,2, 3, 1).to('cpu', torch.uint8).numpy()
    for n,arr in enumerate(split(ndarr,64)):
        #print(squeeze(arr).shape)
        im = Image.fromarray(squeeze(arr))
        im.save(folder+"/"+str(n+start)+".png")
    return n+start


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 3072), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD