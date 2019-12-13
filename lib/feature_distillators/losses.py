import torch
import torch.nn.functional as F

from lib.utils.utils import auto_change_dir


def parse_distillation_loss(args):
    fields = args.distillation.split(",")
    method = fields[0]
    d_args = dict([i.split("-") for i in fields[1:]])

    for k, v in d_args.items():
        if k in ["lambda"]:
            d_args[k] = float(v)
        elif k in ["p"]:
            d_args[k] = int(v)

    print("Perdida", method, "con parametros", d_args)
    losses_list = [hint, att_max, att_mean, PKT, nst_gauss, nst_linear, nst_poly]

    d = dict([(func.__name__, func) for func in losses_list])

    # folder: -> [dataset]/[teacher]/students/[student_model]/[distilation type]/[]
    auto_change_dir(args.distillation.replace(",", "/"))

    try:
        loss = d[method]
    except:
        raise ModuleNotFoundError("Loss not found")

    try:
        return loss(**d_args)
    except:
        raise NameError("There is an argument error")


"""
  Paper: FITNETS: HINTS FOR THIN DEEP NETS
  Code: https://github.com/adri-romsor/FitNets (theano)
"""


def hint():
    def hint_loss(teacher_features, student_features):
        return torch.nn.MSELoss()(teacher_features, student_features)

    return hint_loss


# def TransformAndDistance(T=None,S=None,d=None):
#  re

"""
  Paper: Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via 
  Attention Transfer
  Code: https://github.com/szagoruyko/attention-transfer (pytorch)
"""


def att_mean(p=2):  # Att(attention):

    def at(x):  # todo: gacer mejor
        return F.normalize(x.pow(p).mean(1).view(x.size(0), -1))

    def attention_loss(teacher_features, student_features):
        return (at(student_features) - at(teacher_features)).pow(2).mean()

    return attention_loss


def att_max():  # Att(attention):

    def at(x):  # todo: gacer mejor

        return F.normalize(x.abs().max(1)[0].view(x.size(0), -1))

    def attention_loss(teacher_features, student_features):
        #print(student_features.shape)
        #print(teacher_features.shape)
        return (at(student_features) - at(teacher_features)).pow(2).mean()

    return attention_loss


"""
Paper: Learning Deep Representations with Probabilistic Knowledge Transfer
Code: https://github.com/passalis/probabilistic_kt
"""


def PKT(epsilon=0.0000001):
    def KDE(Tensor):  # Kernel Density Estimation, Stolen from the original code
        # Normalize each vector by its norm
        output_net_norm = torch.sqrt(torch.sum(Tensor ** 2, dim=1, keepdim=True))
        Tensor = Tensor / (output_net_norm + epsilon)
        Tensor[Tensor != Tensor] = 0
        # Calculate the cosine similarity
        similarity = torch.mm(Tensor, Tensor.transpose(0, 1))
        # Scale cosine similarity to 0..1
        similarity = (similarity + 1.0) / 2.0
        # Transform them into probabilities
        return similarity / torch.sum(similarity, dim=1, keepdim=True)

    def divergence(teacher, model):
        target_similarity = KDE(teacher.view(teacher.size(0), -1))
        model_similarity = KDE(model.view(model.size(0), -1))
        return torch.mean(target_similarity * torch.log((target_similarity + epsilon) / (model_similarity + epsilon)))

    def pkt_loss(teacher_features, student_features):  # no free variable should be in declaration.
        return divergence(teacher_features, student_features)

    return pkt_loss


"""
paper: Like What You Like: Knowledge Distill via Neuron Selectivity Transfer
Code: https://github.com/TuSimple/neuron-selectivity-transfer/ (mxnet)
"""


def NST_base(Kernel):
    def MMD(Ft, Fs):
        #floasprint(1, Ft, 2, Fs)
        # FT and FS normalization

        Ft = Ft.view(Ft.shape[0], Ft.shape[1], -1)
        Ft = F.normalize(Ft, dim=0, p=2)

        Fs = Fs.view(Fs.shape[0], Fs.shape[1], -1)
        Fs = F.normalize(Fs, dim=0, p=2)
        # Kernel calculation
        #print(Kernel(Ft, Ft).mean(1), Kernel(Fs, Fs).mean(1) ,(2 * Kernel(Ft, Fs).mean(1)))
        return Kernel(Ft, Ft).mean(1) + Kernel(Fs, Fs).mean(1) - (2 * Kernel(Ft, Fs).mean(1))

    def nst_loss(teacher_features, student_features):
        return torch.mean(MMD(teacher_features, student_features))

    return nst_loss


def nst_linear():
    def Kernel(x, y):
        # x {b,c,w,h}

        return torch.matmul(x, y.transpose(-2, -1)).view(x.shape[0], -1)

    return NST_base(Kernel)


def nst_poly(d=2, c=0):
    def Kernel(x, y):
        # x {b,c,w,h}
        return (torch.matmul(x, y.transpose(-2, -1)).view(x.shape[0], -1) + c).pow(d)

    return NST_base(Kernel)


def nst_gauss():
    def Kernel(x, y):
        pw_dist = torch.add(x.unsqueeze(1), -y.unsqueeze(2)).norm(dim=-1).view(x.shape[0], -1)
        sigma = pw_dist.mean(-1, keepdims=True)
        return torch.exp(-pw_dist / (2 * sigma))

    return NST_base(Kernel)


"""
Paper: A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning
Code: (not original) https://github.com/sseung0703/KD_methods_with_TF
TODO: Requiere inicialiacion, Fome
"""


def FSP():
    raise NotImplementedError
