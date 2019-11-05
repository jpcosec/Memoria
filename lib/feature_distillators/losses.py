import torch
import torch.nn.functional as F
"""
  Paper: FITNETS: HINTS FOR THIN DEEP NETS
"""

def feature_loss(alpha=1):
  def hint_loss(teacher_features, student_features):

    return alpha*torch.nn.MSELoss()(teacher_features,student_features)

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

def att_loss(): #Att(attention):

  def at(x):#todo: gacer mejor
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

  def attention_loss(teacher_features,student_features):
    return (at(student_features) - at(teacher_features)).pow(2).mean()

  return attention_loss