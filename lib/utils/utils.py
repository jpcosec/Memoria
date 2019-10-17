import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

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
from lib.models.mnist_net import MnistNet

def auto_change_dir(path):
  print("Moving to", path)#todo: log
  for folder in path.split("/"):
    if not os.path.exists(folder):
      print("Creating", folder)
      os.mkdir(folder)
    os.chdir(folder)

""" Dataset STUFF"""
def load_cifar10(args):
    auto_change_dir("Cifar10")

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

def load_mnist(args):
    # Load MNIST
    auto_change_dir("Mnist")

    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),  # ToTensor does min-max normalization.
    ]), )

    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),  # ToTensor does min-max normalization.
    ]), )

    # Create DataLoader
    dataloader_args = dict(shuffle=True, batch_size=args.train_batch_size, num_workers=2)
    train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_data, **dataloader_args)

    return train_loader, test_loader, range(10)

def sample(loader):  # Deprecated
    data, target = next(iter(loader))
    data, target = Variable(data.cuda()), Variable(target.cuda())
    return data, target

""" Model Stuff"""

def get_model(model_name):

    if model_name.split("_")[0] == "linear":
        auto_change_dir("linear")
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
                      GoogLeNet=GoogLeNet(),
                      MnistNet=MnistNet()
                      )
    try:
        return model_list[model_name]
    except:
        raise ModuleNotFoundError("Model not found")


def register_hooks(net, idxs, feature):
  """
  Registers a hook in the module of the net
  :param net:
  :param idxs:
  :param feature:
  :return:
  """

  def hook(m, i, o):
    feature[m] = o

  for name, module in net._modules.items():
    for id, layer in enumerate(module.children()):
      if id in idxs:
        layer.register_forward_hook(hook)