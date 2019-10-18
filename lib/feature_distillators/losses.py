import torch
def feature_loss():
  def hint_loss(teacher_features, student_features):
    return torch.nn.PairwiseDistance(p=2)(teacher_features, student_features)

  return hint_loss
