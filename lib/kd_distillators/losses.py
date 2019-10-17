import torch.nn as nn
import torch.nn.functional as F


"""
loss_dict = {"student_scores": S_y_pred, "teacher_scores": T_y_pred, "targets": targets}
"""

def parse_distillation_loss(st):
    fields = st.split(",")
    method = fields[0]
    args = dict([i.split("-") for i in fields[1:]])
    for k, v in args.items():
        args[k] = float(v)

    print("Perdida", method, "con parametros", args)
    losses_list=[KD, KD_CE]
    d = dict([(func.__name__, func) for func in losses_list])
    try:
        loss = d[method]
    except:
        raise ModuleNotFoundError("Loss not found")

    try:
        return loss(**args)
    except:
        raise NameError("There is an argument error")


def KD(T=8):
    """
    "soft label" distillation as proposed by Hinton and Dean in "Distilling the Knowledge in a Neural Network" (2015)
    :param T: Temperature of the distillation
    :return: Loss function
    """

    def KD_loss(input, teacher_logits, T=T):
        return nn.KLDivLoss()(F.log_softmax(input / T, dim=1), F.softmax(teacher_logits / T, dim=1))

    return KD_loss


def KD_CE(alpha=0.5, T=8):
    """
    "soft label" + "hard label" distillation as proposed by Hinton and Dean in "Distilling the Knowledge in a Neural Network" (2015)
    :param T: Temperature of the distillation
    :param alpha: Balance between distillation and Cross Entropy
    :return: Loss function
    """

    def KD_CE_loss(input, teacher_logits, target, alpha=alpha, T=T):
        KD_loss = nn.KLDivLoss()(F.log_softmax(input / T, dim=1),
                                 F.softmax(teacher_logits / T, dim=1))

        CE_loss = nn.CrossEntropyLoss()(input.squeeze(), target)

        return CE_loss * alpha * T * T + KD_loss * (1 - alpha)

    return KD_CE_loss


