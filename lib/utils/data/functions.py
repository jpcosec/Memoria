import numpy as np
import torch
from torch.autograd import Variable


def add_noise(t):
    def tensor_random_noise(input):
        return input + (torch.rand_like(input) * t)
    return tensor_random_noise


def random_return(image):
    #print(image.getpixel((1, 1)))
    return np.random.randint(256,size=(32, 32, 3),dtype=np.uint8)


def sample(loader):  # Deprecated
    data, target = next(iter(loader))
    data, target = Variable(data.cuda()), Variable(target.cuda())
    return data, target




