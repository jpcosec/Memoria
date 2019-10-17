import torch
def feature_loss():
  def hint_loss(teacher_features, student_features,regresor):
    r=regresor(student_features)
    return torch.nn.MSELoss()(teacher_features,r)

  return hint_loss
