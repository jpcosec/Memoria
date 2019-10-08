import os

import torch

import torchvision
import torchvision.transforms as transforms

from lib.utils import progress_bar
from lib.teacher_models.resnet import ResNet18, ResNet101, ResNet50
from lib.teacher_models.mobilenet import MobileNet
from lib.teacher_models.mobilenetv2 import MobileNetV2
from lib.teacher_models.resnext import ResNeXt29_32x4d
from lib.teacher_models.vgg import VGG
from lib.teacher_models.densenet import DenseNet121
from lib.teacher_models.preact_resnet import PreActResNet18
from lib.teacher_models.dpn import DPN92
from lib.teacher_models.senet import SENet18
from lib.teacher_models.efficientnet import EfficientNetB0
from lib.teacher_models.googlenet import GoogLeNet
from lib.dist_model import linear_model


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


# Training
def train(net, epoch, writer, flatten=False):
  global best_acc, trainloader, device, criterion, optimizer
  print('\rEpoch: %d' % epoch)
  net.train()
  train_loss = 0
  correct = 0
  total = 0
  for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    if flatten:
      outputs = net(inputs.view(-1, 3072))
    else:
      outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
    train_acc = 100. * correct / total
    progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                 % (train_loss / (batch_idx + 1), train_acc, correct, total))
    writer.add_scalar('train/loss', train_loss)
    writer.add_scalar('train/acc', train_acc)


def test(net, epoch, writer, flatten=False):
  global best_acc, testloader, device, criterion
  net.eval()
  test_loss = 0
  correct = 0
  total = 0
  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
      inputs, targets = inputs.to(device), targets.to(device)
      if flatten:
        outputs = net(inputs.view(-1, 3072))
      else:
        outputs = net(inputs)
      loss = criterion(outputs, targets)

      test_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()
      writer.add_scalar('test/loss', test_loss)
      progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                   % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

  # Save checkpoint.
  acc = 100. * correct / total
  writer.add_scalar('test/acc', acc)
  if acc > best_acc:
    print('Saving..')
    state = {
      'net': net.state_dict(),
      'acc': acc,
      'epoch': epoch
    }
    if not os.path.isdir('checkpoint'):
      os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.pth')
    best_acc = acc


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


def load_model(args):



  net = get_model(args.model)
  # Model
  print('==> Building model..')

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

  return net, best_acc, start_epoch
