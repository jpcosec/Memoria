import os

import torch
import torch.backends.cudnn as cudnn

from lib.models.resnet import ResNet18, ResNet101, ResNet50
from lib.models.mobilenet import MobileNet
from lib.models.mobilenetv2 import MobileNetV2
from lib.models.resnext import ResNeXt29_32x4d
from lib.models.vgg import VGG
from lib.models.densenet import DenseNet121
from lib.models.preact_resnet import PreActResNet18
from lib.models.dpn import DPN92
from lib.models.senet import SENet18
from lib.models.efficientnet import EfficientNetB0
from lib.models.googlenet import GoogLeNet
from lib.models.linear import linear_model


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
def train(exp,epoch):# TODO: meter accuracy y weas en exp

  print('\rEpoch: %d' % epoch)
  exp.net.train()
  total_loss = 0
  correct = 0
  total = 0
  for batch_idx, (inputs, targets) in enumerate(exp.trainloader):
    inputs, targets = inputs.to(exp.device), targets.to(exp.device)
    exp.optimizer.zero_grad()

    if exp.flatten:
      outputs = exp.net(inputs.view(-1, 3072))
    else:
      outputs = exp.net(inputs)

    loss = exp.criterion(outputs, targets)
    loss.backward()
    exp.optimizer.step()

    total_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

    train_acc  = 100. * correct / total
    train_loss =  total_loss / (batch_idx + 1)

    #progress_bar(batch_idx, len(exp.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #             % (train_loss / (batch_idx + 1), train_acc, correct, total))

    exp.writer.add_scalar('train/loss', train_loss)
    exp.writer.add_scalar('train/acc', train_acc)


def test(exp,epoch):

  exp.net.eval()
  test_loss = 0
  correct = 0
  total = 0
  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(exp.testloader):
      inputs, targets = inputs.to(exp.device), targets.to(exp.device)
      if exp.flatten:
        outputs = exp.net(inputs.view(-1, 3072))
      else:
        outputs = exp.net(inputs)
      loss = exp.criterion(outputs, targets)

      test_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()
      exp.writer.add_scalar('test/loss', test_loss)
      #progress_bar(batch_idx, len(exp.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
      #             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

      # Save checkpoint.
      acc = 100. * correct / total
      exp.writer.add_scalar('test/acc', acc)

  if acc > exp.best_acc:
    print('Saving..')
    state = {
      'net': exp.net.state_dict(),
      'acc': acc,
      'epoch': epoch
    }
    if not os.path.isdir('checkpoint'):
      os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.pth')
    exp.best_acc = acc




def load_model(args,device):
  best_acc = 0  # best test accuracy
  start_epoch = 0  # start from epoch 0 or last checkpoint epoch
  # Model
  print('==> Building model..')
  net = get_model(args.model)
  net = net.to(device)
  if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

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