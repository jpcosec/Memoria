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
            d[field] = kwargs[field]
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
    add_field("student_layer",5)
    add_field("teacher_layer",26)

    args.resume=True
    return args


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Using device", device)  # todo: cambiar a logger
    folder="exp2"
    args = fake_arg()
    #trainloader, testloader, classes = load_cifar10(args)
    #teacher = load_teacher(args, device)

    blocs ={"ResNet101": [26,56,219,239],#Completar
            "MobileNet": [6,15,26,55]
             }
    for student in [ "MobileNet"]:#todo: terminar
        for distillation in ["hint", "att_max", "att_mean", "PKT", "nst_linear", "nst_poly"]:
            for layer,(s_layer,t_layer) in enumerate(zip(blocs["MobileNet"],blocs["ResNet101"])):
                #os.chdir("/home/jp/Memoria/repo/Cifar10/ResNet101/"+folder)#funcionalizar

                #arg = fake_arg(distillation=distillation, student=student,layer=layer,student_layer=s_layer,teacher_layer=t_layer)

                print("python feat_distillation.py --distillation=%s --layer=%i --student=MobileNet --student_layer=%i --teacher_layer=%i"%(distillation,layer,s_layer,t_layer))
                #print("TRAINING-%s-%s-%i"%(student,distillation,layer))
                #experiment_run(arg, device, teacher, testloader, trainloader)
