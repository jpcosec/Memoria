'''Train CIFAR10 with PyTorch.'''

import argparse

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from lib.kd_distillators.losses import parse_distillation_loss
from lib.kd_distillators.utils import *
from lib.utils.data.cifar10 import load_samples
from lib.utils.records_collector_deprecated import maj_key
import json


def experiment_run(args, device, teacher, testloader, trainloader):

    student, best_acc, start_epoch = load_student(args, device)


    criterion = parse_distillation_loss(args)
    eval_criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=args.lr)

    flatten = args.student.split("_")[0] == "linear"
    writer = SummaryWriter("tb_logs")
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
    print("training from epoch",start_epoch, "to", args.epochs)
    for epoch in range(start_epoch, args.epochs):
        exp.train_epoch()
        exp.test_epoch()
    exp.save_model()

def fake_arg(**kwargs):
    args = argparse.Namespace()
    d = vars(args)

    def add_field(field,default):
        if field in kwargs:
            d[field] = kwargs[field]
        else:
            d[field] = default

    add_field('lr' ,0.01)
    add_field('epochs' ,50)
    add_field('train_batch_size' ,128)
    add_field('test_batch_size' ,100)
    add_field('student' ,"ResNet18")
    add_field('teacher' ,"ResNet101")
    add_field('distillation' ,"KD,T-8")


    args.resume=True
    return args



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Using device", device)  # todo: cambiar a logger
    args = fake_arg()
    print(args)
    #trainloader, testloader, classes = load_cifar10(args)
    trainloader, testloader, classes = load_samples(args, "/home/jp/Memoria/repo/Cifar10/VAE_SAMP")
    teacher = load_teacher(args, device)



    for student in ["ResNet18", "MobileNet"]:
        for distillation in ["KD", "KD_CE"]:
            for T in [str(i) for i in [1, 5, 10,50, 100, 1000]]:
                os.chdir("/home/jp/Memoria/repo/Cifar10/ResNet101/exp8"
                         "")#funcionalizar
                dist = distillation+",T-"+ T
                arg = fake_arg(distillation=dist, student=student)

                try:
                    with open("students/"+student+"/"+distillation+"/T-"+T+'/record.json', 'r') as fp:
                        record = json.load(fp)
                        e=maj_key(record["test"])
                        print(e)
                        if e >=arg.epochs:
                            continue
                except:
                    print("hayproblemo")

                print("TRAINING")
                experiment_run(arg, device, teacher, testloader, trainloader)
