'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

import os
import argparse

from lib.teacher_models.resnet import ResNet18 as teacherNetGenerator
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# EfficientNetB0
from lib.utils import progress_bar


# Training
def train(epoch,writer):
    print('\rEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_acc=100.*correct/total
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), train_acc, correct, total))
        writer.add_scalar('train/loss', train_loss)
        writer.add_scalar('train/acc', train_acc)

def test(epoch,writer):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            writer.add_scalar('test/loss', test_loss)
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
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



if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
  parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
  parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint',)
  parser.add_argument('--epochs', default=500, action='store_true', help='total number of epochs to train', )
  args = parser.parse_args()

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  best_acc = 0  # best test accuracy
  start_epoch = 0  # start from epoch 0 or last checkpoint epoch
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
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  # Model
  print('==> Building model..')

  net = teacherNetGenerator()
  net = net.to(device)
  if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True



  if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    if start_epoch >= args.epochs:
      print("Number of epochs")


  writer = SummaryWriter(comment="teacher_trainer")

  criterion = nn.CrossEntropyLoss()
  #optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
  optimizer = optim.Adam(net.parameters() , lr=args.lr)
  for epoch in range(start_epoch, args.epochs):
      train(epoch,writer)
      test(epoch,writer)