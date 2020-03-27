from __future__ import print_function

import argparse
import os

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision.utils import save_image

from lib.utils.funcs import auto_change_dir
from lib.Artificial_Dataset_generators.Autoencoders.utils import load_dataset
from lib.Artificial_Dataset_generators.Autoencoders.Deprecated.conv_VAE import VAE

os.chdir("../../../Cifar10")
print(os.getcwd())

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=802, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--folder',  default="VAE-Dataset",
                    help='output folder')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
print(device)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


test_loader , train_loader = load_dataset(args,kwargs)


auto_change_dir(args.folder)
auto_change_dir("results")
os.chdir("..")





model = VAE().to(device)

assert os.path.isdir("checkpoint"):  # todo: cambiar a non initialized
    # Load checkpoint.
    print('==> Resuming from checkpoint..')

    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']



def main():

    with torch.no_grad():
        for i in range(args.n_samples//64):
            sample = torch.randn(64, 128).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 3, 32, 32),
                       'results/sample.png')

if __name__ == "__main__":
    main()


# © 2019 GitHub, Inc.

# def calc(h_in,kernel_size,stride=1,output_padding=0,padding=0,dilation=1):
#    return (h_in-1)* stride-2*padding + dilation*(kernel_size-1)+output_padding+1
