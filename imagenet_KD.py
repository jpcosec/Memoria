import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torchvision.datasets.folder import default_loader

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from lib.kd_distillators.losses import parse_distillation_loss
from lib.kd_distillators.utils import *
from lib.utils.Imagenet.Dataset import get_imageNet


best_prec1 = 0
os.chdir("Imagenet")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint', )  # change to restart
parser.add_argument('--epochs', default=200, type=int, help='total number of epochs to train')
parser.add_argument('--batch_size', default=100, type=int, help='batch size on train')
parser.add_argument('--student', default="MobileNetV2",
                    help="default ResNet18, other options are VGG, ResNet50, ResNet101, MobileNet, MobileNetV2, "
                         "ResNeXt29, DenseNet, PreActResNet18, DPN92, SENet18, EfficientNetB0, GoogLeNet, "
                         "ShuffleNetG2, ShuffleNetV2 or linear_laysize1,laysize2,laysizen")
parser.add_argument('--teacher', default="ResNet151",
                    help="default ResNet18, other options are VGG, ResNet50, ResNet101, MobileNet, MobileNetV2, "
                         "ResNeXt29, DenseNet, PreActResNet18, DPN92, SENet18, EfficientNetB0, GoogLeNet, "
                         "ShuffleNetG2, ShuffleNetV2 or linear_laysize1,laysize2,laysizen")
parser.add_argument('--distillation', default="KD,T-8.0",
                    help="default=T-3.5, chose one method from lib.kd_distillators an put the numerical params "
                         "separated by , using - instead of =.")
parser.add_argument("--transform", default="none,", help="ej. noise,0.1")
parser.add_argument("--dataset", default="ImageNet,", help="ej. vae_sample")
parser.add_argument("--exp_name", default="test_mobilenet", help='Where to run the experiments')
args = parser.parse_args()

def maim():


    global args
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parser.parse_args()

    teacher = models.resnet152(pretrained=True, progress=True, )
    teacher = teacher.to(device)

    for param in teacher.parameters():
        param.requires_grad = False

    if device == 'cuda':
        teacher = torch.nn.DataParallel(teacher)


    student = models.mobilenet_v2(pretrained=False)
    student = torch.nn.DataParallel(student).cuda()

    trainset,testset=get_imageNet()
    trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=args.batch_size, shuffle=True,num_workers=0)
    testloader = torch.utils.data.DataLoader(testset,
        batch_size=args.batch_size, shuffle=False,num_workers=0)

    teacher.eval()
    student.train()



    #rgs.exp_name is not None:
    # os.chdir("/home/jp/Memoria/repo/Cifar10/ResNet101/") #Linux
    #os.chdir("test")  # Windows
    auto_change_dir(args.exp_name)

    best_acc=0
    start_epoch=0

    criterion = parse_distillation_loss(args)  # CD a distillation
    eval_criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=args.lr)

    flatten = args.student.split("_")[0] == "linear"

    writer = SummaryWriter("tb_logs")
    exp = DistillationExperiment(device=device,  # Todo mover arriba
                                 student=student,
                                 teacher=teacher,
                                 optimizer=optimizer,
                                 criterion=criterion,
                                 eval_criterion=eval_criterion,
                                 linear=flatten,
                                 writer=writer,
                                 testloader=testloader,
                                 trainloader=trainloader,
                                 best_acc=best_acc,
                                 args=args
                                 )

    for epoch in range(start_epoch, args.epochs):

        exp.train_epoch()
        exp.test_epoch()
    exp.save_model()

# TODO Add optimizer to model saves


if __name__ == '__main__':
    maim()
