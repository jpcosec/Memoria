from __future__ import print_function

import argparse
import os

import torch
import torch.utils.data


from lib.utils.funcs import check_folders
from lib.Artificial_Dataset_generators.DCVAE_CIFAR10.utils import save_samples
from lib.Artificial_Dataset_generators.DCVAE_CIFAR10.conv_VAE import VAE

os.chdir("../../../Cifar10")
print(os.getcwd())

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epoch', type=int, default=100, metavar='N',
                    help='epoch to sample')
parser.add_argument('--n_samples', type=int, default=100000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--folder',  default="VAE-Dataset",
                    help='output folder')
args = parser.parse_args()


device = torch.device("cuda" if args.cuda else "cpu")

check_folders(["samples"])


""" Load dataset
"""
model = VAE().to(device)

assert os.path.isdir("checkpoint")  # todo: cambiar a non initialized
# Load checkpoint.
print('==> Resuming from checkpoint..')

checkpoint = torch.load('./checkpoints/ckpt%i.pth'%args.epoch)
model.load_state_dict(checkpoint['net'])
start_epoch = checkpoint['epoch']



def main():

    with torch.no_grad():
        for i in range(args.n_samples//args.batch_size):
            sample = torch.randn(args.batch_size, 128).to(device)
            sample = model.decode(sample).cpu()
            save_samples(sample.view(args.batch_size, 3, 32, 32),
                        start=i*args.batch_size,
                         batch_size=args.batch_size)

if __name__ == "__main__":
    main()


# © 2019 GitHub, Inc.

# def calc(h_in,kernel_size,stride=1,output_padding=0,padding=0,dilation=1):
#    return (h_in-1)* stride-2*padding + dilation*(kernel_size-1)+output_padding+1
