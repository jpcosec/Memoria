

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