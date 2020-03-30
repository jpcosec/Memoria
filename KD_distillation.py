'''Train CIFAR10 with PyTorch.'''

import argparse

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from lib.kd_distillators.losses import parse_distillation_loss
from lib.kd_distillators.utils import *
from lib.utils.utils import cifar10_parser


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Using device", device)  # todo: cambiar a logger

    trainloader, testloader, classes = cifar10_parser(args)# CD a teacher
    teacher = load_teacher(args, device)


    if args.exp_name is not None:
        #os.chdir("/home/jp/Memoria/repo/Cifar10/ResNet101/") #Linux
        os.chdir("C:/Users/PC/PycharmProjects/Memoria/Cifar10/ResNet101/")#Windows
        auto_change_dir(args.exp_name)


    student, best_acc, start_epoch = load_student(args, device)# CD a student

    criterion = parse_distillation_loss(args)#  CD a distillation
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

    for epoch in range(start_epoch, args.epochs):
        exp.train_epoch()
        exp.test_epoch()
    exp.save_model()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint', )  # change to restart
    parser.add_argument('--epochs', default=50, type=int, help='total number of epochs to train')
    parser.add_argument('--train_batch_size', default=128, type=int, help='batch size on train')
    parser.add_argument('--test_batch_size', default=128, type=int, help='batch size on test')
    parser.add_argument('--student', default="ResNet18",
                        help="default ResNet18, other options are VGG, ResNet50, ResNet101, MobileNet, MobileNetV2, "
                             "ResNeXt29, DenseNet, PreActResNet18, DPN92, SENet18, EfficientNetB0, GoogLeNet, "
                             "ShuffleNetG2, ShuffleNetV2 or linear_laysize1,laysize2,laysizen")
    parser.add_argument('--teacher', default="ResNet101",
                        help="default ResNet18, other options are VGG, ResNet50, ResNet101, MobileNet, MobileNetV2, "
                             "ResNeXt29, DenseNet, PreActResNet18, DPN92, SENet18, EfficientNetB0, GoogLeNet, "
                             "ShuffleNetG2, ShuffleNetV2 or linear_laysize1,laysize2,laysizen")
    parser.add_argument('--distillation', default="KD,T-4.0",
                        help="default=T-3.5, chose one method from lib.kd_distillators an put the numerical params "
                             "separated by , using - instead of =.")
    parser.add_argument("--transform", default="none,", help="ej. noise,0.1")
    parser.add_argument("--dataset", default="cifar10,", help="ej. vae_sample")
    parser.add_argument("--exp_name", default=None, help='Where to run the experiments')
    arg = parser.parse_args()

    main(arg)
