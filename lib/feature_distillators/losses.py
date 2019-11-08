import torch
import torch.nn.functional as F

from lib.utils.utils import auto_change_dir


def parse_distillation_loss(args):
  fields = args.distillation.split(",")
  method = fields[0]
  args = dict([i.split("-") for i in fields[1:]])

  for k, v in args.items():
    if k in ["lambda"]:
      args[k] = float(v)
    elif k in["p"]:
      args[k]=int(v)


  print("Perdida", method, "con parametros", args)
  losses_list = [fitnets_loss, att_max,att_mean]

  d = dict([(func.__name__, func) for func in losses_list])

  # folder: -> [dataset]/[teacher]/students/[student_model]/[distilation type]/[]
  auto_change_dir(args.distillation.replace(",","/"))
  try:
    loss = d[method]
  except:
    raise ModuleNotFoundError("Loss not found")

  try:
    return loss(**args)
  except:
    raise NameError("There is an argument error")



"""
  Paper: FITNETS: HINTS FOR THIN DEEP NETS
  
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
  Git-hub: https://github.com/szagoruyko/attention-transfer
"""

#def sum_of_absolutes(activation,p=None):
#  torch.sum(torch.abs(activation),dim=-1)

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


