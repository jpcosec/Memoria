
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data.dataloader as dataloader
from torchvision import transforms
from torchvision.datasets import CIFAR10



def teacher_train(net, train_loader, optimizer, device, n_epochs=100):


    net.train()
    losses = []
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Get Samples
            data=data.to(device)
            target=target.to(device)
            # Init
            optimizer.zero_grad()

            # Predict
            y_pred = net(data)

            # Calculate loss
            loss = F.cross_entropy(y_pred, target)
            losses.append(loss)
            # Backpropagation
            loss.backward()
            optimizer.step()


            # Display
            if batch_idx % 100 == 1:
                print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),

                    100. * batch_idx / len(train_loader),
                    loss),
                    end='')




def acc_test(net, loader, lim=15, flatten=False,):#Todo, generalizar
  d = []
  for batch_idx, (data, target) in enumerate(loader):
    data, target = Variable(data.cuda()), Variable(target.cuda())

    if flatten:
      output = net(data.view(-1, 3072))# 32*32*3
    else:
      output = net(data)
    pred = output.data.max(1)[1]
    d.extend(pred.eq(target).cpu())
    if batch_idx > lim:
      continue

  d = np.array(d)
  accuracy = float(np.sum(d)) / float(d.shape[0])
  return accuracy



def get_dataloaders(data_folder):
  # Load CIFAR10

  train_data = CIFAR10(data_folder, train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),  # ToTensor does min-max normalization.
  ]), )

  test_data = CIFAR10(data_folder, train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),  # ToTensor does min-max normalization.
  ]), )

  # Create DataLoader
  dataloader_args = dict(shuffle=True, batch_size=64, num_workers=1, pin_memory=True)
  train_loader = dataloader.DataLoader(train_data, **dataloader_args)
  test_loader = dataloader.DataLoader(test_data, **dataloader_args)


  return train_loader, test_loader


def sample(loader):
  data, target = next(iter(loader))
  data, target = Variable(data.cuda()), Variable(target.cuda())
  return data, target

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f