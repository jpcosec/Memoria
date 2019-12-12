
'''Train CIFAR10 with PyTorch.'''

import argparse
from torchsummary import summary
import torch
from lib.kd_distillators.utils import silent_load

class FeatureInspector:

  def __init__(self, **kwargs):
    self.teacher = kwargs['teacher']
    self.student = kwargs['student']
    self.device = kwargs['device']

    #print(summary(self.teacher,(3,32,32)))
    #print(summary(self.student, (3, 32, 32)))


    self.teacher_features = {}
    i=0
    for name, module in self.teacher._modules.items():
      print("Teacher Network..", name)
      for block in module.children():
        for layer in block.children():
          print(" block id....",i, layer)
          i+=1
          #def hook(m, i, o):
          #  self.teacher_features[m] = o
          #block.register_forward_hook(hook)

    print(self.teacher.layers)

    self.student_features = {}

    for name, module in self.student._modules.items():
      print("Student Network..", name)
      for id, block in enumerate(module.children()):
        #print("block id....", id, block)
        def hook(m, i, o):
          self.student_features[m] = o
        block.register_forward_hook(hook)

    inp = torch.rand(128, 3, 32, 32).to(self.device)

    self.teacher.eval()
    self.student.eval()

    _ = self.teacher(inp)
    _ = self.student(inp)

    s_sizes =[tensor.shape for tensor in  list(self.student_features.values())]
    t_sizes=[tensor.shape for tensor in  list(self.teacher_features.values())]

    print(t_sizes)
    print(s_sizes)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Using device", device)  # todo: cambiar a logger

    #trainloader, testloader, classes = load_cifar10(args)
    teacher = silent_load(args.teacher, device)
    student = silent_load(args.student, device)
    flatten = args.student.split("_")[0] == "linear"
    layer = args.layer

    exp = FeatureInspector(device=device,
                           student=student,
                           teacher=teacher,
                           linear=flatten,
                           use_regressor=args.distillation=="hint",
                           args = args
                           )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint', )  # change to restart
    parser.add_argument('--epochs', default=100, type=int, help='total number of epochs to train')
    parser.add_argument('--pre', default=50, type=int, help='total number of epochs to train')
    parser.add_argument('--train_batch_size', default=128, type=int, help='batch size on train')
    parser.add_argument('--test_batch_size', default=100, type=int, help='batch size on test')
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
    parser.add_argument('--last_layer', default="KD-CE",
                        help="")
    parser.add_argument("--layer",type=int,default= 5)# Arreglar para caso multicapa
    arg = parser.parse_args()

    main(arg)
