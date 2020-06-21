import argparse
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.models as models
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from lib.kd_distillators.losses import parse_distillation_loss
from lib.kd_distillators.utils import *
from lib.utils.imagenet.Dataset import get_dataloaders
from lib.utils.imagenet.utils import load_model


def main(args):


    #global args
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parser.parse_args()

    trainloader, testloader = get_dataloaders(args.batch_size)
    auto_change_dir(args.exp_name)

    teacher=load_model(args.teacher, trainable=False,device=device)
    student = load_model(args.student)

    teacher.eval()
    student.train()



    #rgs.exp_name is not None:
    # os.chdir("/home/jp/Memoria/repo/Cifar10/ResNet101/") #Linux
    #os.chdir("test")  # Windows


    best_acc=0
    start_epoch=0

    criterion = parse_distillation_loss(args)  # CD a distillation
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

# TODO Add optimizer to model saves


if __name__ == '__main__':
    best_prec1 = 0

    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint', )  # change to restart
    parser.add_argument('--epochs', default=100, type=int, help='total number of epochs to train')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size on train')
    parser.add_argument('--student', default="ResNet18",
                        help="default ResNet18, other options are VGG, ResNet50, ResNet101, MobileNet, MobileNetV2, "
                             "ResNeXt29, DenseNet, PreActResNet18, DPN92, SENet18, EfficientNetB0, GoogLeNet, "
                             "ShuffleNetG2, ShuffleNetV2 or linear_laysize1,laysize2,laysizen")
    parser.add_argument('--teacher', default="ResNet152",
                        help="default ResNet18, other options are VGG, ResNet50, ResNet101, MobileNet, MobileNetV2, "
                             "ResNeXt29, DenseNet, PreActResNet18, DPN92, SENet18, EfficientNetB0, GoogLeNet, "
                             "ShuffleNetG2, ShuffleNetV2 or linear_laysize1,laysize2,laysizen")
    parser.add_argument('--distillation', default="KD,T-8.0",
                        help="default=T-3.5, chose one method from lib.kd_distillators an put the numerical params "
                             "separated by , using - instead of =.")
    parser.add_argument("--transform", default="none,", help="ej. noise,0.1")
    parser.add_argument("--dataset", default="ImageNet,", help="ej. vae_sample")
    parser.add_argument("--exp_name", default="ultimors", help='Where to run the experiments')
    args = parser.parse_args()

    main(args)
