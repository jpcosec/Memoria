import torch
from torch import nn as nn
from torch.nn.functional import one_hot


class Generator(nn.Module):

  def __init__(self):
    super(Generator, self).__init__()

    self.layer0 = nn.Sequential(nn.Linear(110, 1024, bias=False),
                                nn.BatchNorm1d(1024),
                                nn.ReLU(True))

    # input 100*1*1
    self.layer1 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 4, 1, 0, bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(True))

    # input 512*4*4
    self.layer2 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(True))
    # input 256*8*8
    self.layer3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                                nn.BatchNorm2d(128),
                                nn.ReLU(True))
    # input 128*16*16
    self.layer4 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(True))
    # input 64*32*32
    self.layer5 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                                nn.Tanh())
    # output 3*64*64

    # self.embedding = nn.Embedding(10,100)

  def forward(self, noise, label):
    label_embedding = one_hot(label, 10)

    # print(label_embedding.shape)
    # print(label_embedding)

    # print(noise.shape)
    x = torch.cat((noise, label_embedding.float()), dim=1)

    # x = x.view(-1, 110, 1, 1)
    x = self.layer0(x)
    x = x.view(-1, 1024, 1, 1)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    return x


class Discriminator(nn.Module):

  def __init__(self):
    super(Discriminator, self).__init__()

    # input 3*64*64
    self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.LeakyReLU(0.2, True),
                                nn.Dropout2d(0.5))

    # input 64*32*32
    self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                                nn.BatchNorm2d(128),
                                nn.LeakyReLU(0.2, True),
                                nn.Dropout2d(0.5))
    # input 128*16*16
    self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                                nn.BatchNorm2d(256),
                                nn.LeakyReLU(0.2, True),
                                nn.Dropout2d(0.5))
    # input 256*8*8
    self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                                nn.BatchNorm2d(512),
                                nn.LeakyReLU(0.2, True))
    # input 512*4*4
    self.validity_layer = nn.Sequential(nn.Conv2d(512, 1, 4, 1, 0, bias=False),
                                        nn.Sigmoid())

    self.label_layer = nn.Sequential(nn.Conv2d(512, 11, 4, 1, 0, bias=False),
                                     nn.LogSoftmax(dim=1))

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    validity = self.validity_layer(x)
    plabel = self.label_layer(x)

    validity = validity.view(-1)
    plabel = plabel.view(-1, 11)

    return validity, plabel
