import torch
def feature_loss(alpha=0.000001):
  def hint_loss(teacher_features, student_features):

    return alpha*torch.nn.MSELoss()(teacher_features,student_features)

  return hint_loss
