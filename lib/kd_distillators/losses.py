import torch.nn as nn
import torch.nn.functional as F
from lib.utils.utils import auto_change_dir


"""
loss_dict = {"input": S_y_pred, "teacher_logits": T_y_pred, "target": targets}
"""

def parse_distillation_loss(args, string_input=False):
    if string_input:
        fields=args.split(",")
    else:
        fields = args.distillation.split(",")
    method = fields[0]
    ar = dict([i.split("-") for i in fields[1:]])
    for k, v in ar.items():
        ar[k] = float(v)

    print("Perdida", method, "con parametros", ar)
    losses_list=[KD, KD_CE]
    d = dict([(func.__name__, func) for func in losses_list])


    try:
        loss = d[method]
    except:
        raise ModuleNotFoundError("Loss not found")
    if not string_input:

        try:
            # folder: -> [dataset]/[teacher]/students/[student_model]/[distilation type]/[]
            auto_change_dir("/".join([args.distillation[:args.distillation.find(",")],
                                      args.distillation[args.distillation.find(",") + 1:]]))
            return loss(**ar)
        except:

            raise NameError("There is an argument error")
    else:
        return loss(**args)#todo: ordenar

def KD(T=8):
    """
    "soft label" distillation as proposed by Hinton and Dean in "Distilling the Knowledge in a Neural Network" (2015)
    :param T: Temperature of the distillation
    :return: Loss function
    """

    def KD_loss(input, teacher_logits):
        return nn.KLDivLoss()(F.log_softmax(input / T, dim=1), F.softmax(teacher_logits / T, dim=1))

    return KD_loss


def KD_CE(alpha=0.5, T=8):
    """
    "soft label" + "hard label" distillation as proposed by Hinton and Dean in "Distilling the Knowledge in a Neural Network" (2015)
    :param T: Temperature of the distillation
    :param alpha: Balance between distillation and Cross Entropy
    :return: Loss function
    """

    def KD_loss(input, teacher_logits):
        return nn.KLDivLoss()(F.log_softmax(input / T, dim=1),
                                 F.softmax(teacher_logits / T, dim=1))

    def CE_loss(input,target):
        return  nn.CrossEntropyLoss()(input.squeeze(), target)

    def KD_CE_loss(input, teacher_logits, target):

        return CE_loss(input,target) * alpha * T * T + KD_loss(input, teacher_logits) * (1 - alpha)

    return KD_CE_loss


