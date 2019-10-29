'''Train CIFAR10 with PyTorch.'''

import argparse

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from lib.kd_distillators.losses import parse_distillation_loss
from lib.kd_distillators.utils import *
from lib.utils.utils import load_cifar10
from lib.utils.records_collector import maj_key


def experiment_run(args, device, teacher, testloader, trainloader):

    student, best_acc, start_epoch = load_student(args, device)
    writer = SummaryWriter("tb_logs")

    criterion = parse_distillation_loss(args.distillation)
    eval_criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=args.lr)

    flatten = args.student.split("_")[0] == "linear"

    exp = DistillationExperiment(device=device,  # Todo mover arriba
                                 student=student,
                                 teacher=teacher,
                                 optimizer=optimizer,
                                 criterion=criterion,
                                 eval_criterion=eval_criterion,
                                 linear=flatten,
                                 writer=writer,
                                 testloader=testloader,
                                 trainloader=trainloader,
                                 best_acc=best_acc,
                                 args=args
                                 )
    if maj_key(exp.record) >= 99:
        print("Already trained")
        return None
    print("training from epoch",start_epoch, "to", args.epochs)
    for epoch in range(start_epoch, args.epochs):
        exp.train_epoch()
        exp.test_epoch()


def fake_arg(**kwargs):
    args = argparse.Namespace()

    if 'lr' in kwargs:
        args.lr = kwargs["lr"]
    else:
        args.lr = 0.1

    if 'epochs' in kwargs:
        args.epochs = kwargs["epochs"]
    else:
        args.epochs = 100

    if 'train_batch_size' in kwargs:
        args.train_batch_size = kwargs["train_batch_size"]
    else:
        args.train_batch_size = 128

    if 'test_batch_size' in kwargs:
        args.test_batch_size = kwargs["test_batch_size"]
    else:
        args.test_batch_size = 100

    if 'student' in kwargs:
        args.student = kwargs["student"]
    else:
        args.student = "ResNet18"

    if 'teacher' in kwargs:
        args.teacher = kwargs["teacher"]
    else:
        args.teacher = "ResNet101"

    if 'distillation' in kwargs:
        args.distillation = kwargs["distillation"]
    else:
        args.distillation = "KD,T-8"

    args.resume=True
    return args


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Using device", device)  # todo: cambiar a logger
    args = fake_arg()
    trainloader, testloader, classes = load_cifar10(args)
    teacher = load_teacher(args, device)

    for student in ["ResNet18", "MobileNet", "EfficientNetB0"]:
        for distillation in ["KD", "KD_CE"]:
            for T in [str(i) for i in [1, 5, 10,50, 100, 1000]]:
                os.chdir("/home/jp/Memoria/repo/Cifar10/ResNet101/exp1")#funcionalizar
                dist = distillation+",T-"+ T
                arg = fake_arg(distillation=dist, student=student)
                experiment_run(arg, device, teacher, testloader, trainloader)
