import os

import torch
import torch.backends.cudnn as cudnn

from lib.utils.utils import get_model


def load_model(args, device):
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    # Model
    print('==> Building model..')
    net = get_model(args.model)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True


    if args.resume:#todo: cambiar a non initialized
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
            print("Number of epochs already trained, requested", args.epochs, "trained",start_epoch)
    else:
        os.mkdir(args.model)# Mover a experiment
        os.chdir(args.model)
    return net, best_acc, start_epoch