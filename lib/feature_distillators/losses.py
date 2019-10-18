import torch
def feature_loss():
  def hint_loss(teacher_features, student_features):
    size=student_features.shape
    #todo promedio print(size)
    return torch.nn.MSELoss()(teacher_features,student_features)

  return hint_loss
