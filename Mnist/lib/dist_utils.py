import torch.nn as nn
import torch.nn.functional as F

def dist_loss_gen(T=8):
  def dist_loss(student_scores, teacher_scores, T=T):
    return nn.KLDivLoss()(F.log_softmax(student_scores / T, dim=1), F.softmax(teacher_scores / T, dim=1))

  return dist_loss