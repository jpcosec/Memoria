'''Train CIFAR10 with PyTorch.'''

import argparse

from torch.utils.tensorboard import SummaryWriter

from lib.feature_distillators.losses import parse_distillation_loss
from lib.feature_distillators.utils import *
from lib.kd_distillators.losses import parse_distillation_loss as last_layer_loss_parser
from lib.kd_distillators.utils import load_student, load_teacher
from lib.utils.utils import load_cifar10, auto_change_dir, random_return, add_noise, load_samples

import os

import torchvision.transforms as transforms
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Using device", device)  # todo: cambiar a logger

    """transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(add_noise(0.1)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])"""

    #trainloader, testloader, classes = load_cifar10(args)
    #trainloader, testloader, classes = load_samples(arg, "/home/jp/Memoria/repo/Cifar10/VAE_SAMP")
    teacher = load_teacher(args, device)

    #todo: arreglar
    os.chdir("/home/jp/Memoria/repo/Cifar10/ResNet101/" + "exp6")

    student, best_acc, start_epoch = load_student(args, device)

    feat_loss =  parse_distillation_loss(args)
    kd_loss=last_layer_loss_parser(args.last_layer,string_input=True)


    eval_criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=args.lr)#todo: evaluar si mover en exp
    flatten = args.student.split("_")[0] == "linear"
    layer = args.layer
    idxs = [layer]
    auto_change_dir(",".join([str(i) for i in idxs]))

    writer = SummaryWriter("tb_logs")#todo mover dentro de exp

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
                            use_regressor=args.distillation=="hint",
                            args = args
                            )
    if exp.epoch+1 < args.epochs:
        print("training",exp.epoch, "-",args.epochs)
        for epoch in range(exp.epoch, args.epochs):
            exp.train_epoch()
            exp.test_epoch()
        exp.save_model()
    else:
        print("epochs surpassed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint', )  # change to restart
    parser.add_argument('--epochs', default=50, type=int, help='total number of epochs to train')
    parser.add_argument('--pre', default=50, type=int, help='total number of epochs to train')
    parser.add_argument('--train_batch_size', default=128, type=int, help='batch size on train')
    parser.add_argument('--test_batch_size', default=128, type=int, help='batch size on test')
    parser.add_argument('--student', default="MobileNet",
                        help="default ResNet18, other options are VGG, ResNet50, ResNet101, MobileNet, MobileNetV2, "
                             "ResNeXt29, DenseNet, PreActResNet18, DPN92, SENet18, EfficientNetB0, GoogLeNet, "
                             "ShuffleNetG2, ShuffleNetV2 or linear_laysize1,laysize2,laysizen")
    parser.add_argument('--teacher', default="ResNet101",
                        help="default ResNet18, other options are VGG, ResNet50, ResNet101, MobileNet, MobileNetV2, "
                             "ResNeXt29, DenseNet, PreActResNet18, DPN92, SENet18, EfficientNetB0, GoogLeNet, "
                             "ShuffleNetG2, ShuffleNetV2 or linear_laysize1,laysize2,laysizen")
    parser.add_argument('--distillation', default="nst_linear",
                        help="feature-alpha")
    parser.add_argument('--last_layer', default="KD,T-8",
                        help="")
    parser.add_argument("--student_layer",type=int,default= 5)# Arreglar para caso multicapa
    parser.add_argument("--teacher_layer", type=int, default=26)  # Arreglar para caso
    parser.add_argument("--layer", type=int, default=2)  # solo para el nombre

    arg = parser.parse_args()

    main(arg)
