'''Train CIFAR10 with PyTorch.'''
import argparse

import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from lib.utils.Experiment import Experiment
from lib.utils.data.cifar10 import load_cifar10
from lib.teacher.utils import *

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("using device", device)

    trainloader, testloader, classes = load_cifar10(args)

    net, best_acc, start_epoch = load_model(args, device)
    print("best acc", best_acc)

    writer = SummaryWriter("teacher_trainer")
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    flatten = args.model.split("_")[0] == "linear"
    exp = Experiment(device=device,
                     net=net,
                     optimizer=optimizer,
                     criterion=criterion,
                     linear=args.model.split("_")[0] == "linear",
                     writer=writer,
                     testloader=testloader,
                     trainloader=trainloader,
                     best_acc=best_acc,
                     start_epoch=start_epoch
                     )

    for epoch in range(start_epoch, args.epochs):
        exp.train_epoch()
        exp.test_epoch()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint', )
    parser.add_argument('--epochs', default=400, type=int, help='total number of epochs to train')
    parser.add_argument('--train_batch_size', default=128, type=int, help='total number of epochs to train')
    parser.add_argument('--test_batch_size', default=100, type=int, help='total number of epochs to train')
    parser.add_argument('--model', default="ResNet18",
                        help="default ResNet18, other options are VGG, ResNet50, ResNet101, MobileNet, MobileNetV2, "
                             "ResNeXt29, DenseNet, PreActResNet18, DPN92, SENet18, EfficientNetB0, GoogLeNet, "
                             "ShuffleNetG2, ShuffleNetV2 or linear_laysize1,laysize2,laysizen")

    arg = parser.parse_args()


    main(arg)