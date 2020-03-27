from __future__ import print_function

import argparse
import os

import torch
import torch.utils.data
from torch import optim
from torchvision.utils import save_image

from lib.utils.funcs import auto_change_dir, check_folders
from lib.Artificial_Dataset_generators.Autoencoders.utils import load_dataset, loss_function
from lib.Artificial_Dataset_generators.Autoencoders.Deprecated.conv_VAE import VAE

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
parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--sample', action='store_true', default=False,
                    help='samples trained model')
parser.add_argument('--folder',  default="VAE-Dataset",
                    help='output folder')

args = parser.parse_args()


args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
print(device)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

test_loader, train_loader = load_dataset(args, kwargs)

auto_change_dir(args.folder)
check_folders()

model = VAE().to(device)

"""if os.path.isdir("checkpoint"):  # todo: cambiar a non initialized
    # Load checkpoint.
    print('==> Resuming from checkpoint..')

    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']
else:
    auto_change_dir("checkpoint")
    os.chdir("..")
    start_epoch = 0"""

optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        # print("oli")
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



def main():
    for epoch in range(0, args.epochs + 1):
        train(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 128).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 3, 32, 32),
                       'outs/sample_' + str(epoch) + '.png')
        state = {
            'net': model.state_dict(),
            'epoch': epoch
        }
        torch.save(state, './checkpoints/ckpt%i.pth'%epoch)


if __name__ == "__main__":
    main()

# © 2019 GitHub, Inc.

# def calc(h_in,kernel_size,stride=1,output_padding=0,padding=0,dilation=1):
#    return (h_in-1)* stride-2*padding + dilation*(kernel_size-1)+output_padding+1
