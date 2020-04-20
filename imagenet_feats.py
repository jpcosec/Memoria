'''Train CIFAR10 with PyTorch.'''

import argparse

from torch.utils.tensorboard import SummaryWriter

from lib.feature_distillators.losses import parse_distillation_loss
from lib.feature_distillators.utils import *
from lib.kd_distillators.losses import parse_distillation_loss as last_layer_loss_parser

import argparse
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from lib.utils.imagenet.Dataset import get_dataloaders
from lib.utils.imagenet.utils import load_model
from lib.utils.funcs import auto_change_dir

import os


def experiment_run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainloader, testloader = get_dataloaders(args.batch_size)

    auto_change_dir(args.exp_name)

    print("Using device", device)  # todo: cambiar a logger
    teacher = load_model(args.teacher, trainable=False, device=device)
    student = load_model(args.student)

    teacher.eval()
    student.train()


    best_acc = 0
    start_epoch = 0

    feat_loss = parse_distillation_loss(args)
    kd_loss = last_layer_loss_parser(args.last_layer, string_input=True)

    eval_criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=args.lr)  # todo: evaluar si mover en exp
    flatten = args.student.split("_")[0] == "linear"
    layer = args.layer
    idxs = [layer]
    auto_change_dir(",".join([str(i) for i in idxs]))

    writer = SummaryWriter("tb_logs")  # todo mover dentro de exp

    exp = FeatureExperiment(device=device,
                            student=student,
                            teacher=teacher,
                            optimizer=optimizer,
                            kd_criterion=kd_loss,
                            ft_criterion=feat_loss,
                            eval_criterion=eval_criterion,
                            linear=flatten,
                            writer=writer,
                            testloader=testloader,
                            trainloader=trainloader,
                            best_acc=best_acc,
                            idxs=idxs,
                            use_regressor=args.distillation == "hint",
                            args=args
                            )
    if exp.epoch + 1 < args.epochs:
        print("training", exp.epoch, "-", args.epochs)
        for epoch in range(exp.epoch, args.epochs):
            exp.train_epoch()
            exp.test_epoch()
        exp.save_model()
    else:
        print("epochs surpassed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Imagenet Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint', )  # change to restart
    parser.add_argument('--epochs', default=100, type=int, help='total number of epochs to train')
    parser.add_argument('--pre', default=50, type=int, help='total number of epochs to train')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size on train and test')
    parser.add_argument('--student', default="MobileNetV2",  # "ResNet18",#
                        help="default ResNet18, other options are VGG, ResNet50, ResNet101, MobileNet, MobileNetV2, "
                             "ResNeXt29, DenseNet, PreActResNet18, DPN92, SENet18, EfficientNetB0, GoogLeNet, "
                             "ShuffleNetG2, ShuffleNetV2 or linear_laysize1,laysize2,laysizen")
    parser.add_argument('--teacher', default="ResNet152",
                        help="default ResNet18, other options are VGG, ResNet50, ResNet101, MobileNet, MobileNetV2, "
                             "ResNeXt29, DenseNet, PreActResNet18, DPN92, SENet18, EfficientNetB0, GoogLeNet, "
                             "ShuffleNetG2, ShuffleNetV2 or linear_laysize1,laysize2,laysizen")
    parser.add_argument("--transform", default="none,", help="ej. noise,0.1")
    parser.add_argument("--dataset", default="ImageNet,", help="ej. vae_sample")
    parser.add_argument("--exp_name", default="exp_", help='Where to run the experiments')
    parser.add_argument('--distillation', default="PKT",
                        help="feature-alpha")
    parser.add_argument('--last_layer', default="KD,T-8",
                        help="")
    parser.add_argument("--student_layer", type=int, default=30)  # Arreglar para caso multicapa
    parser.add_argument("--teacher_layer", type=int, default=39)  # Arreglar para caso
    parser.add_argument("--layer", type=int, default=1)  # cambiar a block
    parser.add_argument("--shape", type=int, default=224)
    arg = parser.parse_args()

    experiment_run(arg)
