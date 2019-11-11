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
    elif k in["p"]:
      d_args[k]=int(v)


  print("Perdida", method, "con parametros", d_args)
  losses_list = [fitnets_loss, att_max,att_mean, PKT]

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

def fitnets_loss():
  def hint_loss(teacher_features, student_features):

    return torch.nn.MSELoss()(teacher_features,student_features)

  return hint_loss

#def TransformAndDistance(T=None,S=None,d=None):
#  re

"""
  Paper: Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via 
  Attention Transfer
  Code: https://github.com/szagoruyko/attention-transfer (pytorch)
"""

def att_mean(p=2): #Att(attention):

  def at(x):#todo: gacer mejor
    return F.normalize(x.pow(p).mean(1).view(x.size(0), -1))

  def attention_loss(teacher_features,student_features):
    return (at(student_features) - at(teacher_features)).pow(2).mean()

  return attention_loss


def att_max(): #Att(attention):

  def at(x):#todo: gacer mejor
    return F.normalize(x.abs().max(1)[0].view(x.size(0),-1))


  def attention_loss(teacher_features,student_features):
    return (at(student_features) - at(teacher_features)).pow(2).mean()

  return attention_loss


"""
Paper: Learning Deep Representations with Probabilistic Knowledge Transfer
Code: https://github.com/passalis/probabilistic_kt
"""

def PKT(epsilon=0.0000001):

  def KDE(Tensor):#Kernel Density Estimation, Stolen from the original code
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
    target_similarity=KDE(teacher.view(teacher.size(0), -1))
    model_similarity=KDE(model.view(model.size(0), -1))
    return torch.mean(target_similarity * torch.log((target_similarity + epsilon) / (model_similarity + epsilon)))

  def pkt_loss(teacher_features, student_features):#no free variable should be in declaration.
    return divergence(teacher_features,student_features)

  return pkt_loss


"""
paper: Like What You Like: Knowledge Distill via Neuron Selectivity Transfer
Code: https://github.com/TuSimple/neuron-selectivity-transfer/ (mxnet)
"""