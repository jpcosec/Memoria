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

import torchvision
from torchvision import transforms as transforms
import os

def get_dataloaders(args,transform_train=None,transform_test=None):

  if transform_train is None:
    transform_train = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      # transforms.Lambda(random_return),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])

  if transform_test is None:
    transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

  trainset = torchvision.datasets.ImageFolder(
    root=args.dataset_folder+"/train/", transform=transform_train)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

  testset = torchvision.datasets.ImageFolder(
    root=args.dataset_folder + "/test/", transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=0)

  return trainloader, testloader


def experiment_run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainloader, testloader = get_dataloaders(args)

    auto_change_dir(args.exp_name)

    print("Using device", device)  # todo: cambiar a logger

    # This will download a model with it's weights. For using another model just instantiate it here.
    teacher = load_model(args.teacher, trainable=False, device=device)
    student = load_model(args.student)

    teacher.eval()
    student.train()


    best_acc = 0
    start_epoch = 0

    feat_loss = parse_distillation_loss(args)
    kd_loss = last_layer_loss_parser(args.log_dist, string_input=True)

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
                            use_regressor=args.feat_dist == "hint",
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
    parser = argparse.ArgumentParser(description='PyTorch feature distillation')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint', )  # change to restart
    parser.add_argument('--epochs', default=100, type=int, help='total number of epochs to train')
    #parser.add_argument('--pre', default=100, type=int, help='total number of epochs to train')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size on train and test')
    parser.add_argument('--student', default="ResNet18",#"MobileNetV2",  # "ResNet18",#
                        help="default ResNet18, use a model of https://pytorch.org/docs/stable/torchvision/models.html")
    parser.add_argument('--teacher', default="ResNet152",
                        help="default ResNet152, use a model of https://pytorch.org/docs/stable/torchvision/models.html")
    #parser.add_argument("--transform", default="none,", help="ej. noise,0.1")
    parser.add_argument("--dataset_folder", help="""
    Folder of the dataset with this structure
    /{dataset_folder}/
        /test
            /Class_0/{001.jpg,xx.jpg}
            /Class_1/{001.jpg,xx.jpg}
            ...
        /train
            /Class_0/{001.jpg,xx.jpg}
            /Class_1/{001.jpg,xx.jpg}
            ...            
    """)
    parser.add_argument("--exp_name", default="test", help='Where to save experiment results')
    parser.add_argument('--feat_dist', default="PKT",help="""
    Feature transform distillation. Choose one from KD, KD_CE, CE,  
    Details in lib/feature_distillators/losses""")
    parser.add_argument('--log_dist', default="KD,T-8",help="""
    Logit distillation. Choose one from hint, att_max, att_mean, PKT, nst_gauss,  
    nst_linear,nst_poly. Details in lib/kd_distillators/losses""")
    parser.add_argument("--student_layer", type=int, default=30,help="""
    Student convolutional layer in wich make the distillation. Check width and height dimensions with torchsummary https://github.com/sksq96/pytorch-summary
    """)  # Arreglar para caso multicapa
    parser.add_argument("--teacher_layer", type=int, default=39,help="""
    Student convolutional layer in wich make the distillation. Check width and height dimensions with torchsummary https://github.com/sksq96/pytorch-summary
    """)  # Arreglar para caso
    parser.add_argument("--layer", type=int, default=1,help="Block, just as reference")  # cambiar a block
    parser.add_argument("--shape", type=int, default=224, help="Image shape, Use only squared images")
    arg = parser.parse_args()

    experiment_run(arg)
