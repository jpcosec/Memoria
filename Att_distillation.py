'''Train CIFAR10 with PyTorch.'''

import argparse

from torch.utils.tensorboard import SummaryWriter

from lib.feature_distillators.losses import att_loss
from lib.feature_distillators.utils import *
from lib.kd_distillators.losses import KD
from lib.kd_distillators.utils import load_student, load_teacher
from lib.utils.utils import load_cifar10


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Using device", device)  # todo: cambiar a logger

    trainloader, testloader, classes = load_cifar10(args)
    teacher = load_teacher(args, device)
    student, best_acc, start_epoch = load_student(args, device)

    writer = SummaryWriter("tb_logs")

    eval_criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=args.lr)

    flatten = args.student.split("_")[0] == "linear"
    arr = args.distillation.split("-")
    alpha = float(arr[1])
    layer = int(arr[2])

    idxs = [layer]

    exp = FeatureExperiment(device=device,  # Todo mover arriba
                            student=student,
                            teacher=teacher,
                            optimizer=optimizer,
                            kd_criterion=KD(T=8.0),
                            ft_criterion=att_loss(),
                            eval_criterion=eval_criterion,
                            linear=flatten,
                            writer=writer,
                            testloader=testloader,
                            trainloader=trainloader,
                            best_acc=best_acc,
                            idxs=idxs,
                            use_regressor=False
                            )

    for epoch in range(start_epoch, args.epochs):
        exp.train_epoch()
        exp.test_epoch()


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
    parser.add_argument('--distillation', default="attention-1-5",
                        help="feature-alpha-layer")
    arg = parser.parse_args()

    main(arg)
