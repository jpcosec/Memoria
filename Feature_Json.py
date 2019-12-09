'''Train CIFAR10 with PyTorch.'''


import argparse
import os
import json

from torch.utils.tensorboard import SummaryWriter

from lib.feature_distillators.losses import parse_distillation_loss
from lib.feature_distillators.utils import *
from lib.kd_distillators.losses import KD
from lib.kd_distillators.utils import load_student, load_teacher
from lib.utils.utils import load_cifar10, auto_change_dir



def experiment_run(args, device, teacher, testloader, trainloader):

    student, best_acc, start_epoch = load_student(args, device)
    writer = SummaryWriter("tb_logs")

    feat_loss =  parse_distillation_loss(args)
    eval_criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=args.lr)

    flatten = args.student.split("_")[0] == "linear"
    layer = args.layer
    idxs = [layer]

    auto_change_dir(",".join([str(i) for i in idxs]))

    exp = FeatureExperiment(device=device,  # Todo mover arriba
                                 student=student,
                                 teacher=teacher,
                                 optimizer=optimizer,
                                 kd_criterion=KD(T=8.0),
                                 ft_criterion=feat_loss,
                                 eval_criterion=eval_criterion,
                                 linear=flatten,
                                 writer=writer,
                                 testloader=testloader,
                                 trainloader=trainloader,
                                 best_acc=best_acc,
                                 idxs=idxs,
                                 use_regressor=args.use_regressor,
                                 args=args
                                 )

    if exp.epoch+1 < args.epochs:
        print("training",exp.epoch, "-",args.epochs)
        for epoch in range(exp.epoch, args.epochs):
            exp.train_epoch()
            exp.test_epoch()

    else:
        print("epochs surpassed")


def fake_arg(**kwargs):
    args = argparse.Namespace()
    d = vars(args)

    def add_field(field,default):
        if field in kwargs:
            d[field] = kwargs["lr"]
        else:
            d[field] = default

    add_field('lr' ,0.1)
    add_field('epochs' ,100)
    add_field('train_batch_size' ,128)
    add_field('test_batch_size' ,100)
    add_field('student' ,"ResNet18")
    add_field('teacher' ,"ResNet101")
    add_field('distillation' ,"nst_linear")
    add_field('last_layer',"KD")
    add_field("layer", 5)# Arreglar para caso multicapa
    add_field('pre',50)

    args.resume=True
    return args


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Using device", device)  # todo: cambiar a logger
    args = fake_arg()
    trainloader, testloader, classes = load_cifar10(args)
    teacher = load_teacher(args, device)



    for student in [ "MobileNet"]:#todo: terminar
        for distillation in ["hint", "att_max", "att_mean", "PKT", "nst_gauss", "nst_linear", "nst_poly"]:
            for layer in [str(i) for i in [1, 5, 10,50, 100, 1000]]:
                os.chdir("/home/jp/Memoria/repo/Cifar10/ResNet101/exp1")#funcionalizar
                dist = distillation+",T-"+ T
                arg = fake_arg(distillation=dist, student=student)

                try:
                    with open("students/"+student+"/"+distillation+"/T-"+T+'/record.json', 'r') as fp:
                        record = json.load(fp)
                        e=maj_key(record["test"])
                        print(e)
                        if e >= 99:
                            continue
                except:
                    print("hayproblemo")

                print("TRAINING")
                experiment_run(arg, device, teacher, testloader, trainloader)
