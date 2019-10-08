import torch.nn as nn
import torch.nn.functional as F
from lib.utils import acc_test, sample


def dist_loss_gen(T=8):
  def dist_loss(student_scores, teacher_scores, T=T):
    return nn.KLDivLoss()(F.log_softmax(student_scores / T, dim=1), F.softmax(teacher_scores / T, dim=1))

  return dist_loss


# class distiller()

def train_epoch(distiller, teacher, loader, device, writer):
  for batch_idx, (data, target) in enumerate(loader):  # 784= 28*28
    x_train = data.to(device)
    y_train = target.to(device)

    distiller["optimizer"].zero_grad()
    # Forward pass

    # Predecir
    S_y_pred = distiller["model"](x_train.view(-1, 784))
    T_y_pred = teacher(x_train)

    # Compute Loss
    loss = distiller["criterion"](S_y_pred, T_y_pred)
    # Backward pass
    loss.backward()
    distiller["optimizer"].step()
    writer.add_scalar('dist-loss/train', loss.detach().numpy())


def test_sample(distiller, teacher, loaders, device, eval_criterion, writer):
  train_loader, test_loader = loaders

  x_train, y_train = sample(train_loader)  # Sample lleva automaticamente los datos a Device

  y_pred = distiller["model"](x_train.view(-1, 3072))
  train_stats = eval_criterion(y_pred, y_train)

  x_test, y_test = sample(test_loader)
  y_pred = distiller["model"](x_test.view(-1, 3072))
  test_stats = eval_criterion(y_pred.squeeze(), y_test)

  y_predT = teacher(x_test)
  test_statsT = eval_criterion(y_predT.squeeze(), y_test)

  # todo JSONEAR O ALGO

  writer.add_scalar('CE/train', train_stats)
  writer.add_scalar('CE/test', test_stats)
  writer.add_scalar('CE/teacher', test_statsT)
  writer.add_scalar('Acc/train', acc_test(distiller["model"], train_loader, flatten=True))
  writer.add_scalar('Acc/test', acc_test(distiller["model"], test_loader, flatten=True))
