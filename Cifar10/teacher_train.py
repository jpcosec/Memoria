'''Train CIFAR10 with PyTorch.'''
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

global device

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from lib.teacher_utils import *

def load_model(args):



  return net, best_acc, start_epoch

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
  parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
  parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint', )
  parser.add_argument('--epochs', default=500, type=int, help='total number of epochs to train')
  parser.add_argument('--train_batch_size', default=128, type=int, help='total number of epochs to train')
  parser.add_argument('--test_batch_size', default=100, type=int, help='total number of epochs to train')
  parser.add_argument('--model', default="ResNet18",
                      help="default ResNet18, other options are VGG, ResNet50, ResNet101, MobileNet, MobileNetV2, ResNeXt29, DenseNet, PreActResNet18, DPN92, SENet18, EfficientNetB0, GoogLeNet, ShuffleNetG2, ShuffleNetV2 or linear_laysize1,laysize2,laysizen")

  args = parser.parse_args()

  global best_acc, trainloader, testloader, criterion, optimizer


  #args=device

  trainloader, testloader, classes = load_dataset(args)

  best_acc = 0  # best test accuracy
  start_epoch = 0  # start from epoch 0 or last checkpoint epoch
  # Model
  print('==> Building model..')
  net = get_model(args.model)
  net = net.to(device)
  if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

  if args.resume:
    assert os.path.isdir(args.model), 'Error: model not initialized'
    os.chdir(args.model)
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    if start_epoch >= args.epochs:
      print("Number of epochs already trained")
  else:
    os.mkdir(args.model)
    os.chdir(args.model)
  #net, best_acc, start_epoch = load_model(args)






  writer = SummaryWriter("teacher_trainer")
  criterion = nn.CrossEntropyLoss()
  # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
  optimizer = optim.Adam(net.parameters(), lr=args.lr)

  for epoch in range(start_epoch, args.epochs):
    flatten_input = args.model.split("_")[0] == "linear"
    train(epoch, writer, flatten=flatten_input)
    test(epoch, writer, flatten=flatten_input)
