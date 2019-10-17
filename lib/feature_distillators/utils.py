
import torch
import torch.backends.cudnn as cudnn

from lib.utils.Experiment import Experiment
from lib.utils.utils import get_model
import os

def register_hooks(net,idxs,feature):
  def hook(m, i, o):
    feature[m] = o 
  for name, module in net._modules.items():
    for id,layer in enumerate(module.children()):
      if id in idxs:  
        layer.register_forward_hook(hook)

def load_teacher(args, device):
  print('==> Building teacher model..', args.teacher)
  net = get_model(args.teacher)
  net = net.to(device)

  for param in net.parameters():
    param.requires_grad = False

  if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

  assert os.path.isdir(args.teacher), 'Error: model not initialized'
  os.chdir(args.teacher)
  # Load checkpoint.
  print('==> Resuming from checkpoint..')
  assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
  checkpoint = torch.load('./checkpoint/ckpt.pth')
  net.load_state_dict(checkpoint['net'])

  return net


def load_student(args, device):
  best_acc = 0  # best test accuracy
  start_epoch = 0  # start from epoch 0 or last checkpoint epoch
  folder = "feature_distilators/" + args.student + "/" + args.distillation
  # Model
  print('==> Building student model..', args.student)
  net = get_model(args.student)
  net = net.to(device)
  if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

  if args.resume:
    assert os.path.isdir(folder), 'Error: model not initialized'
    os.chdir(folder)
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
    if not os.path.isdir("feature_distilators/"):
      os.mkdir("feature_distilators")

    if not os.path.isdir("feature_distilators/" + args.student):
      os.mkdir("feature_distilators/" + args.student)
    os.mkdir(folder)
    os.chdir(folder)