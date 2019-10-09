'''Train CIFAR10 with PyTorch.'''
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from lib.utils import experiment, load_dataset

###global device

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from lib.student.utils import *



if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
  parser.add_argument("--temp", type=float, default=3.5)
  parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
  parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint', )
  parser.add_argument('--epochs', default=500, type=int, help='total number of epochs to train')
  parser.add_argument('--train_batch_size', default=128, type=int, help='total number of epochs to train')
  parser.add_argument('--test_batch_size', default=100, type=int, help='total number of epochs to train')
  parser.add_argument('--student', default="ResNet101",
                      help="default ResNet18, other options are VGG, ResNet50, ResNet101, MobileNet, MobileNetV2, ResNeXt29, DenseNet, PreActResNet18, DPN92, SENet18, EfficientNetB0, GoogLeNet, ShuffleNetG2, ShuffleNetV2 or linear_laysize1,laysize2,laysizen")
  parser.add_argument('--teacher', default="ResNet101",
                      help="default ResNet18, other options are VGG, ResNet50, ResNet101, MobileNet, MobileNetV2, ResNeXt29, DenseNet, PreActResNet18, DPN92, SENet18, EfficientNetB0, GoogLeNet, ShuffleNetG2, ShuffleNetV2 or linear_laysize1,laysize2,laysizen")

  args = parser.parse_args()



  trainloader, testloader, classes = load_dataset(args)
  teacher=load_teacher(args,device)
  student, best_acc, start_epoch = load_student(args,device)
  writer = SummaryWriter("student_trainer")

  criterion = dist_loss_gen(args.temp)
  eval_criterion = torch.nn.CrossEntropyLoss()
  optimizer = optim.Adam(student.parameters(), lr=args.lr)

  flatten=args.student.split("_")[0] == "linear"

  exp=distillation_experiment(device=device,
           student=student,
           teacher=teacher,
           optimizer=optimizer,
           criterion=criterion,
           linear=flatten,
           writer=writer,
           testloader=testloader,
           trainloader=trainloader,
           best_acc=best_acc
           )

  for epoch in range(start_epoch, args.epochs):
    train(exp,epoch)
    test(exp,epoch)
