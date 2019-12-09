'''Train CIFAR10 with PyTorch.'''

import argparse

from torch.utils.tensorboard import SummaryWriter

from lib.feature_distillators.losses import parse_distillation_loss
from lib.feature_distillators.utils import *
from lib.kd_distillators.losses import KD_CE
from lib.kd_distillators.utils import load_student, load_teacher
from lib.utils.utils import load_cifar10, auto_change_dir


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Using device", device)  # todo: cambiar a logger

    trainloader, testloader, classes = load_cifar10(args)
    teacher = load_teacher(args, device)
    student, best_acc, start_epoch = load_student(args, device)
    feat_loss =  parse_distillation_loss(args)



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
                            kd_criterion=KD_CE(T=8.0),
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

    else:
        print("epochs surpassed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint', )  # change to restart
    parser.add_argument('--epochs', default=100, type=int, help='total number of epochs to train')
    parser.add_argument('--pre', default=50, type=int, help='total number of epochs to train')
    parser.add_argument('--train_batch_size', default=128, type=int, help='batch size on train')
    parser.add_argument('--test_batch_size', default=100, type=int, help='batch size on test')
    parser.add_argument('--student', default="ResNet18",
                        help="default ResNet18, other options are VGG, ResNet50, ResNet101, MobileNet, MobileNetV2, "
                             "ResNeXt29, DenseNet, PreActResNet18, DPN92, SENet18, EfficientNetB0, GoogLeNet, "
                             "ShuffleNetG2, ShuffleNetV2 or linear_laysize1,laysize2,laysizen")
    parser.add_argument('--teacher', default="ResNet101",
                        help="default ResNet18, other options are VGG, ResNet50, ResNet101, MobileNet, MobileNetV2, "
                             "ResNeXt29, DenseNet, PreActResNet18, DPN92, SENet18, EfficientNetB0, GoogLeNet, "
                             "ShuffleNetG2, ShuffleNetV2 or linear_laysize1,laysize2,laysizen")
    parser.add_argument('--distillation', default="nst_linear",
                        help="feature-alpha")
    parser.add_argument('--last_layer', default="KD-CE",
                        help="")
    parser.add_argument("--layer",type=int,default= 5)# Arreglar para caso multicapa
    arg = parser.parse_args()

    main(arg)
