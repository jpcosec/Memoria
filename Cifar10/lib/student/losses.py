import torch.nn as nn
import torch.nn.functional as F

def parse_distillation_loss(st):
  fields=st.split(",")
  method=fields[0]
  args=dict([i.split("-") for i in fields[1:]])
  for k, v in args.items():
    args[k] = float(v)

  print("Perdida",method, "con parametros",args)
  d=dict(soft=soft,
         composed=composed)
  try:
    loss=d[method]
  except:
    raise ModuleNotFoundError("Loss not found")

  try:
    return loss(**args)
  except:
    raise NameError("There is an argument error")


def soft(T=8):
  """
  "soft label" distillation as proposed by Hinton and Dean in "Distilling the Knowledge in a Neural Network" (2015)
  :param T: Temperature of the distillation
  :return: Loss function
  """
  def dist_loss(student_scores, teacher_scores, T=T):
    return nn.KLDivLoss()(F.log_softmax(student_scores / T, dim=1), F.softmax(teacher_scores / T, dim=1))

  return dist_loss


def composed(alpha=0.5, T=8):
  """
  "soft label" + "hard label" distillation as proposed by Hinton and Dean in "Distilling the Knowledge in a Neural Network" (2015)
  :param T: Temperature of the distillation
  :param alpha: Balance between distillation and Cross Entropy
  :return: Loss function
  """
  def total_loss(student_scores, teacher_scores, y, alpha=alpha, T=T):
    KD_loss = nn.KLDivLoss()(F.log_softmax(student_scores / T, dim=1),
                             F.softmax(teacher_scores / T, dim=1))

    CE_loss = nn.CrossEntropyLoss()(student_scores.squeeze(), y)

    return CE_loss * alpha * T * T + KD_loss * (1 - alpha)

  return total_loss
