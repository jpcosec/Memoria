import os

import torch
import torch.backends.cudnn as cudnn

from lib.utils import get_model


# Training
def train(exp,epoch):
  print('\rTraining epoch: %d' % exp.epoch)
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

    acc  = 100. * correct / total
    total_loss =  total_loss / (batch_idx + 1)

    #progress_bar(batch_idx, len(exp.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #             % (train_loss / (batch_idx + 1), train_acc, correct, total))

    logs = {'loss': total_loss,
            'acc': acc}

    exp.record_step(logs)

  exp.record_epoch(logs, acc)


def test(exp,epoch):
  print('\rTesting epoch: %d' % exp.epoch)
  exp.net.eval()
  total_loss = 0
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

      total_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()

      #progress_bar(batch_idx, len(exp.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
      #             % (total_loss / (batch_idx + 1), 100. * correct / total, correct, total))

      # Save checkpoint.
      acc = 100. * correct / total

      logs={'loss': total_loss,
            'acc': acc}
      exp.record_step(logs, test=True)

    exp.record_epoch(logs, acc, test=True)




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