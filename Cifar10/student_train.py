'''Train CIFAR10 with PyTorch.'''
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from lib.utils import load_dataset

###global device



from lib.student.utils import *
from lib.student.losses import parse_distillation_loss


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
  parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
  parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint', )
  parser.add_argument('--epochs', default=500, type=int, help='total number of epochs to train')
  parser.add_argument('--train_batch_size', default=128, type=int, help='total number of epochs to train')
  parser.add_argument('--test_batch_size', default=100, type=int, help='total number of epochs to train')
  parser.add_argument('--student', default="ResNet18",
                      help="default ResNet18, other options are VGG, ResNet50, ResNet101, MobileNet, MobileNetV2, ResNeXt29, DenseNet, PreActResNet18, DPN92, SENet18, EfficientNetB0, GoogLeNet, ShuffleNetG2, ShuffleNetV2 or linear_laysize1,laysize2,laysizen")
  parser.add_argument('--teacher', default="ResNet101",
                      help="default ResNet18, other options are VGG, ResNet50, ResNet101, MobileNet, MobileNetV2, ResNeXt29, DenseNet, PreActResNet18, DPN92, SENet18, EfficientNetB0, GoogLeNet, ShuffleNetG2, ShuffleNetV2 or linear_laysize1,laysize2,laysizen")
  parser.add_argument('--distillation',default="soft,T-3.5",
                      help="default=soft,T-3.5, chose one method from lib.student an put the numerical params separated by , using - instead of =.")
  args = parser.parse_args()

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  print("Using device", device)# todo: cambiar a logger

  trainloader, testloader, classes = load_dataset(args)
  teacher=load_teacher(args,device)
  student, best_acc, start_epoch = load_student(args,device)
  writer = SummaryWriter("tb_logs")#todo: solucionar webeo de los steps

  criterion = parse_distillation_loss(args.distillation)
  eval_criterion = torch.nn.CrossEntropyLoss()
  optimizer = optim.Adam(student.parameters(), lr=args.lr)

  flatten=args.student.split("_")[0] == "linear"

  exp=distillation_experiment( device=device,#Todo mover arriba
                               student=student,
                               teacher=teacher,
                               optimizer=optimizer,
                               criterion=criterion,
                               eval_criterion=eval_criterion,
                               linear=flatten,
                               writer=writer,
                               testloader=testloader,
                               trainloader=trainloader,
                               best_acc=best_acc
                               )

  for epoch in range(start_epoch, args.epochs):
    train(exp,epoch)
    test(exp,epoch)
    exp.update_epoch()
