
import torch
import torch.utils.data

from torch import nn

def calc(h_in,kernel_size,stride=1,output_padding=0,padding=0,dilation=1):
    return (h_in-1)* stride-2*padding + dilation*(kernel_size-1)+output_padding+1

def conv_output_shape(h, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    h = floor( ((h + (2 * pad) - ( dilation * (kernel_size - 1) ) - 1 )/ stride) + 1)
    return h

class Encoder(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.decoder = self._make_decoder()
        self.decoder0 = nn.Sequential(nn.Linear(10, 128),
                                      nn.ReLU(True),
                                      nn.Linear(128, 8192),
                                      nn.ReLU(True))


    def _make_decoder(self):
        cfg = [(32, 1), (32, 1)]
        layers = []
        in_channels = 32

        for x, s in cfg:
            layers += [nn.ConvTranspose2d(in_channels, x,kernel_size=3, stride=2,output_padding=1,padding=2),
                       nn.ReLU(True)]
            in_channels = x
        layers += [nn.ConvTranspose2d(in_channels, 32,kernel_size=3,padding=1,output_padding=1,stride=2),
                   nn.ReLU(True)]

        layers += [nn.Conv2d(32, 3, kernel_size=3, stride=1,padding=1)]

        return nn.Sequential(*layers)

    def encode(self, x):

        out = self.encoder(x)
        out = out.view(out.size(0), -1)
        out = self.encoder0(out)
        return self.mu(out), self.logvar(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = self._make_encoder()
        self.encoder0 = nn.Linear(8192, 128)

        self.mu = nn.Linear(128, 128)
        self.logvar = nn.Linear(128, 128)

        self.decoder = self._make_decoder()
        self.decoder0 = nn.Sequential(nn.Linear(128, 128),
                                      nn.ReLU(True),
                                      nn.Linear(128, 512),
                                      nn.ReLU(True),
                                      nn.Linear(512, 8192),
                                      nn.ReLU(True))

    def _make_encoder(self):
        cfg = [(3, 1), (32, 2), (32, 1), (32, 1)]
        layers = []
        in_channels = 3

        for x, s in cfg:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, stride=s,padding=1),
                       # nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True)]
            in_channels = x

        return nn.Sequential(*layers)

    def _make_decoder(self):
        cfg = [(64, 1), (64, 1),(64, 1)]
        layers = []
        in_channels = 32

        for x, s in cfg:
            layers += [nn.ConvTranspose2d(in_channels, x, kernel_size=3, stride=s,padding=1),
                       nn.ReLU(True),
                       nn.Conv2d(x, x, kernel_size=3,padding=1),
                       # nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(x, x, kernel_size=3,padding=1),
                       # nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True)
                       ]
            in_channels = x
        layers += [nn.ConvTranspose2d(in_channels, 32,kernel_size=3,padding=1,output_padding=1,stride=2),
                   nn.ReLU(True)]

        layers += [nn.Conv2d(32, 3, kernel_size=3, stride=1,padding=1)]

        return nn.Sequential(*layers)

    def encode(self, x):

        out = self.encoder(x)
        out = out.view(out.size(0), -1)
        out = self.encoder0(out)

        return self.mu(out), self.logvar(out)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        out = self.decoder0(z)
        out = out.view(out.size(0), 32, 16, 16)
        out = self.decoder(out)
        return torch.sigmoid(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar

