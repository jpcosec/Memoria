from __future__ import print_function

import argparse
import os

import torch
import torch.utils.data

from lib.utils.funcs import check_folders, auto_change_dir
from lib.Artificial_Dataset_generators.DCVAE_CIFAR10.utils import save_samples
from lib.Artificial_Dataset_generators.ACGAN_cifar10.model import Generator

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
parser.add_argument('--folder', default="GAN-Dataset",
                    help='output folder')

args = parser.parse_args()
args = parser.parse_args()

device = torch.device("cuda")
auto_change_dir(args.folder)
check_folders(["samples"])

""" Load dataset
"""
gen = Generator().to(device)
assert os.path.isdir("checkpoints")  # todo: cambiar a non initialized
# Load checkpoint.
print('==> Resuming from checkpoint..')

checkpoint = torch.load('./checkpoints/ckpt%i.pth' % args.epoch)
gen.load_state_dict(checkpoint['net'])
start_epoch = checkpoint['epoch']


def main():
  labels = torch.arange(0, 100, dtype=torch.long, device=device) // 10
  with torch.no_grad():
    for i in range(args.n_samples // 100):
      noise = torch.randn(100, 100, device=device)
      gen_images = gen(noise, labels).cpu()
      images = images / 2 + 0.5
      save_samples(gen_images,
                   start=i * args.batch_size,
                   batch_size=args.batch_size)


if __name__ == "__main__":
  main()

# © 2019 GitHub, Inc.

# def calc(h_in,kernel_size,stride=1,output_padding=0,padding=0,dilation=1):
#    return (h_in-1)* stride-2*padding + dilation*(kernel_size-1)+output_padding+1
