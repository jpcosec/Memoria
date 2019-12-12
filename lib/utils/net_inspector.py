
'''Train CIFAR10 with PyTorch.'''

from collections import OrderedDict
import argparse
#from torchsummary import summary
import torch
from lib.kd_distillators.utils import silent_load
import torch.nn as nn
import numpy as np



class FeatureInspector:

  def __init__(self, **kwargs):
    self.teacher = kwargs['teacher']
    self.student = kwargs['student']
    self.device = kwargs['device']

    #print(summary(self.teacher,(3,32,32)))
    #print(summary(self.student, (3, 32, 32)))



    self.teacher_features = {}

    def summary(model, input_size, batch_size=-1, device="cuda"):

      def register_hook(module):

        def hook(module, input, output):
          class_name = str(module.__class__).split(".")[-1].split("'")[0]
          module_idx = len(summary)

          m_key = "%s-%i" % (class_name, module_idx + 1)
          summary[m_key] = OrderedDict()
          summary[m_key]["input_shape"] = list(input[0].size())
          summary[m_key]["input_shape"][0] = batch_size
          if isinstance(output, (list, tuple)):
            summary[m_key]["output_shape"] = [
              [-1] + list(o.size())[1:] for o in output
            ]
          else:
            summary[m_key]["output_shape"] = list(output.size())
            summary[m_key]["output_shape"][0] = batch_size

          params = 0
          if hasattr(module, "weight") and hasattr(module.weight, "size"):
            params += torch.prod(torch.LongTensor(list(module.weight.size())))
            summary[m_key]["trainable"] = module.weight.requires_grad
          if hasattr(module, "bias") and hasattr(module.bias, "size"):
            params += torch.prod(torch.LongTensor(list(module.bias.size())))
          summary[m_key]["nb_params"] = params
          summary[m_key]["module"] = module

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
          hooks.append(module.register_forward_hook(hook))

      device = device.lower()
      assert device in [
        "cuda",
        "cpu",
      ], "Input device is not valid, please specify 'cuda' or 'cpu'"

      if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
      else:
        dtype = torch.FloatTensor

      # multiple inputs to the network
      if isinstance(input_size, tuple):
        input_size = [input_size]

      # batch_size of 2 for batchnorm
      x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
      # print(type(x[0]))

      # create properties
      summary = OrderedDict()
      hooks = []

      # register hook
      model.apply(register_hook)

      # make a forward pass
      # print(x.shape)
      model(*x)

      # remove these hooks
      for h in hooks:
        h.remove()

      print("----------------------------------------------------------------")
      line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
      print(line_new)
      print("================================================================")
      total_params = 0
      total_output = 0
      trainable_params = 0
      for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
          layer,
          str(summary[layer]["output_shape"]),
          "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
          if summary[layer]["trainable"] == True:
            trainable_params += summary[layer]["nb_params"]
        print(line_new)

      # assume 4 bytes/number (float on cuda).
      total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
      total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
      total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
      total_size = total_params_size + total_output_size + total_input_size

      print("================================================================")
      print("Total params: {0:,}".format(total_params))
      print("Trainable params: {0:,}".format(trainable_params))
      print("Non-trainable params: {0:,}".format(total_params - trainable_params))
      print("----------------------------------------------------------------")
      print("Input size (MB): %0.2f" % total_input_size)
      print("Forward/backward pass size (MB): %0.2f" % total_output_size)
      print("Params size (MB): %0.2f" % total_params_size)
      print("Estimated Total Size (MB): %0.2f" % total_size)
      print("----------------------------------------------------------------")
      return summary

    print(summary(self.teacher,(3,32,32)))

    """
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


    self.student_features = {}

    for name, module in self.student._modules.items():
      print("Student Network..", name)
      for id, block in enumerate(module.children()):
        #print("block id....", id, block)
        def hook(m, i, o):
          self.student_features[m] = o
        block.register_forward_hook(hook)


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


    inp = torch.rand(128, 3, 32, 32).to(self.device)

    self.teacher.eval()
    self.student.eval()

    _ = self.teacher(inp)
    _ = self.student(inp)

    s_sizes =[tensor.shape for tensor in  list(self.student_features.values())]
    t_sizes=[tensor.shape for tensor in  list(self.teacher_features.values())]

    print(t_sizes)
    print(s_sizes)
  """


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
