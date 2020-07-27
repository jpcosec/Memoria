'''Test torchvision model with imagenet'''
import argparse

import torch.nn as nn
#import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from lib.kd_distillators.utils import *

from lib.utils.Experiment import Experiment
from lib.teacher.utils import *
from lib.utils.imagenet.Dataset import get_dataloaders
from lib.utils.imagenet.utils import load_model

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("using device", device)

    #trainloader, testloader, classes = load_cifar10(args)

    trainloader, testloader = get_dataloaders(args.batch_size)
    auto_change_dir(args.exp_name)

    net=load_model(args.teacher, trainable=False,device=device)
    best_acc = 0
    start_epoch = 0

    writer = SummaryWriter("teacher_trainer")
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = None# optim.Adam(net.parameters(), lr=args.lr)


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

    exp.test_epoch()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    #parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint', )
    #parser.add_argument('--epochs', default=400, type=int, help='total number of epochs to train')
    parser.add_argument('--batch_size', default=128, type=int, help='total number of epochs to train')
    #parser.add_argument('--test_batch_size', default=100, type=int, help='total number of epochs to train')
    parser.add_argument('--model', default="ResNet18",
                        help="default ResNet18, other options are VGG, ResNet50, ResNet101, MobileNet, MobileNetV2, "
                             "ResNeXt29, DenseNet, PreActResNet18, DPN92, SENet18, EfficientNetB0, GoogLeNet, "
                             "ShuffleNetG2, ShuffleNetV2 or linear_laysize1,laysize2,laysizen")
    parser.add_argument('--last_layer', default="CE")
    parser.add_argument("--dataset", default="ImageNet,", help="ej. vae_sample")
    parser.add_argument("--exp_name", default="ultimors", help='Where to run the experiments')
    arg = parser.parse_args()


    main(arg)