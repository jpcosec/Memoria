from __future__ import print_function

import argparse
import os

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

os.chdir("../../../Cifar10")
print(os.getcwd())

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=400, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
print(device)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
transformc = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          #transforms.Lambda(random_return),
          transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

      ])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transformc)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)


testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transformc)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)

os.chdir('VAE_FC')


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        cfg=[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        rcfg = [512, 512, 'M', 512, 512, 'M', 256, 256, 'M', 128, 'M', 64]

        self.encoder = self._make_layers(cfg)

        self.mu = nn.Linear(512, 128)
        self.logvar = nn.Linear(512, 128)



        self.decoder0 = nn.Linear(128, 512)

        self.decoder=self._make_layers(rcfg, encoder=False)

    def _make_layers(self, cfg, encoder=True):

        layers = []
        in_channels = 3
        if not encoder:
            in_channels=128
        for x in cfg:
            if x == 'M' and encoder:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'M' and not encoder:
                layers += [nn.MaxPool2d(kernel_size=1, stride=1,dilation=2)]
            else:

                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        if encoder:
            layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        else:
            layers += nn.Conv2d(in_channels, 3, kernel_size=3, padding=1,dilation=2)
        return nn.Sequential(*layers)

    def encode(self, x):

        out = self.encoder(x)
        out = out.view(out.size(0), -1)
        return self.mu(out), self.logvar(out)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        out = self.decoder0(z)
        out = out.view(out.size(0),2,2, -1)
        out = self.decoder(out)
        return torch.sigmoid(self.decoder3(out))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 3072), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        print("oli")
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(args.batch_size, 3, 32, 32)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 32).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 3, 32, 32),
                       'results/sample_' + str(epoch) + '.png')
        state = {
            'net': model.state_dict(),
            'epoch':epoch
        }
        torch.save(state, './checkpoint/ckpt.pth')
# © 2019 GitHub, Inc.
