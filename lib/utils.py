

from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch

from lib.models.densenet import DenseNet121
from lib.models.dpn import DPN92
from lib.models.efficientnet import EfficientNetB0
from lib.models.googlenet import GoogLeNet
from lib.models.linear import linear_model
from lib.models.mobilenet import MobileNet
from lib.models.mobilenetv2 import MobileNetV2
from lib.models.preact_resnet import PreActResNet18
from lib.models.resnet import ResNet18, ResNet50, ResNet101
from lib.models.resnext import ResNeXt29_32x4d
from lib.models.senet import SENet18
from lib.models.vgg import VGG


def load_dataset(args):
  # Data
  print('==> Preparing data..')
  transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  return trainloader, testloader, classes




def sample(loader):# Deprecated
  data, target = next(iter(loader))
  data, target = Variable(data.cuda()), Variable(target.cuda())
  return data, target

# '''Some helper functions for PyTorch, including:
#     - get_mean_and_std: calculate the mean and std value of dataset.
#     - msr_init: net parameter initialization.
#     - progress_bar: progress bar mimic xlua.progress.
#
# import os
# import sys
# import time
# import math
#
# import torch.nn as nn
# import torch.nn.init as init
#
#
# def get_mean_and_std(dataset):
#     #'''#Compute the mean and std value of dataset.'''
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
#     mean = torch.zeros(3)
#     std = torch.zeros(3)
#     print('==> Computing mean and std..')
#     for inputs, targets in dataloader:
#         for i in range(3):
#             mean[i] += inputs[:,i,:,:].mean()
#             std[i] += inputs[:,i,:,:].std()
#     mean.div_(len(dataset))
#     std.div_(len(dataset))
#     return mean, std
#
# def init_params(net):
#     '''Init layer parameters.'''
#     for m in net.modules():
#         if isinstance(m, nn.Conv2d):
#             init.kaiming_normal(m.weight, mode='fan_out')
#             if m.bias:
#                 init.constant(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm2d):
#             init.constant(m.weight, 1)
#             init.constant(m.bias, 0)
#         elif isinstance(m, nn.Linear):
#             init.normal(m.weight, std=1e-3)
#             if m.bias:
#                 init.constant(m.bias, 0)
#
#
# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)
#
# TOTAL_BAR_LENGTH = 65.
# last_time = time.time()
# begin_time = last_time
# def progress_bar(current, total, msg=None):
#     global last_time, begin_time
#     if current == 0:
#         begin_time = time.time()  # Reset for new bar.
#
#     cur_len = int(TOTAL_BAR_LENGTH*current/total)
#     rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
#
#     sys.stdout.write(' [')
#     for i in range(cur_len):
#         sys.stdout.write('=')
#     sys.stdout.write('>')
#     for i in range(rest_len):
#         sys.stdout.write('.')
#     sys.stdout.write(']')
#
#     cur_time = time.time()
#     step_time = cur_time - last_time
#     last_time = cur_time
#     tot_time = cur_time - begin_time
#
#     L = []
#     L.append('  Step: %s' % format_time(step_time))
#     L.append(' | Tot: %s' % format_time(tot_time))
#     if msg:
#         L.append(' | ' + msg)
#
#     msg = ''.join(L)
#     sys.stdout.write(msg)
#     for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
#         sys.stdout.write(' ')
#
#     # Go back to the center of the bar.
#     for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
#         sys.stdout.write('\b')
#     sys.stdout.write(' %d/%d ' % (current+1, total))
#
#     if current < total-1:
#         sys.stdout.write('\r')
#     else:
#         sys.stdout.write('\n')
#     sys.stdout.flush()
#
# def format_time(seconds):
#     days = int(seconds / 3600/24)
#     seconds = seconds - days*3600*24
#     hours = int(seconds / 3600)
#     seconds = seconds - hours*3600
#     minutes = int(seconds / 60)
#     seconds = seconds - minutes*60
#     secondsf = int(seconds)
#     seconds = seconds - secondsf
#     millis = int(seconds*1000)
#
#     f = ''
#     i = 1
#     if days > 0:
#         f += str(days) + 'D'
#         i += 1
#     if hours > 0 and i <= 2:
#         f += str(hours) + 'h'
#         i += 1
#     if minutes > 0 and i <= 2:
#         f += str(minutes) + 'm'
#         i += 1
#     if secondsf > 0 and i <= 2:
#         f += str(secondsf) + 's'
#         i += 1
#     if millis > 0 and i <= 2:
#         f += str(millis) + 'ms'
#         i += 1
#     if f == '':
#         f = '0ms'
#     return f
# '''

def get_model(model_name):
  if model_name.split("_")[0] == "linear":
    shape = [int(st) for st in model_name.split("_")[1].split(",")]
    return linear_model(shape)

  model_list = dict(VGG=VGG('VGG19'),
                    ResNet18=ResNet18(),
                    ResNet50=ResNet50(),
                    ResNet101=ResNet101(),
                    MobileNet=MobileNet(),
                    MobileNetV2=MobileNetV2(),
                    ResNeXt29=ResNeXt29_32x4d(),
                    DenseNet=DenseNet121(),
                    PreActResNet18=PreActResNet18(),
                    DPN92=DPN92(),
                    SENet18=SENet18(),
                    EfficientNetB0=EfficientNetB0(),
                    GoogLeNet=GoogLeNet(), )
  try:
    return model_list[model_name]
  except:
    raise ModuleNotFoundError("Model not found")