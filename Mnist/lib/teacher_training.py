
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import MNIST

from Teacher_model import Net


def train_net(net,train_loader):


    net.train()
    losses = []
    for epoch in range(100):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Get Samples
            data, target = Variable(data.cuda()), Variable(target.cuda())
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
    save_model=True
    if (save_model):
        torch.save(net.state_dict(),PATH)




def test(net, loader, lim=10):
  d = []
  for batch_idx, (data, target) in enumerate(loader):
    data, target = Variable(data.cuda()), Variable(target.cuda())

    output = net(data)
    pred = output.data.max(1)[1]
    d.extend(pred.eq(target).cpu())
    if batch_idx > lim:
      continue

  d = np.array(d)
  accuracy = float(np.sum(d)) / float(d.shape[0])
  return accuracy






