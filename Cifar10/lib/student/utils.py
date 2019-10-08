import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def dist_loss_gen(T=8):
  def dist_loss(student_scores, teacher_scores, T=T):
    return nn.KLDivLoss()(F.log_softmax(student_scores / T, dim=1), F.softmax(teacher_scores / T, dim=1))

  return dist_loss


# class distiller()

def train_epoch(exp,epoch):
  print('\rEpoch: %d' % epoch)
  exp.student.train()
  exp.trainer.train()
  total_loss = 0
  correct = 0
  total = 0
  for batch_idx, (inputs, targets) in enumerate(exp.trainloader):  # 784= 28*28
    inputs, targets = inputs.to(exp.device), targets.to(exp.device)
    exp.optimizer.zero_grad()

    # Predecir
    if exp.flatten:
      S_y_pred = exp.student(inputs.view(-1, 3072))
    else:
      S_y_pred = exp.student(inputs)

    T_y_pred = exp.teacher(inputs)

    # Compute Loss
    loss = exp.criterion(S_y_pred, T_y_pred)
    # Backward pass
    loss.backward()
    exp.optimizer.step()

    total_loss += loss.item()
    _, predicted = S_y_pred.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

    train_acc = 100. * correct / total
    train_loss = total_loss / (batch_idx + 1)

    train_EC = exp.eval_criterion(S_y_pred, targets).item()

    exp.writer.add_scalar('train/loss', train_loss)
    exp.writer.add_scalar('train/acc', train_acc)
    exp.writer.add_scalar("train/EvalCriterion",train_EC)




def test_sample(exp,epoch):
  exp.student.eval()
  exp.teacher.eval()
  test_loss = 0
  correct = 0
  total = 0

  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(exp.testloader):

      # Predecir
      if exp.flatten:
        S_y_pred = exp.student(inputs.view(-1, 3072))
      else:
        S_y_pred = exp.student(inputs)

      loss = exp.criterion(S_y_pred, targets)
      T_y_pred = exp.teacher(inputs)

      student_eval = exp.eval_criterion(S_y_pred.squeeze(), targets)
      teacher_eval = exp.eval_criterion(T_y_pred.squeeze(), targets)

      test_loss += loss.item()
      _, predicted = S_y_pred.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()

      #progress_bar(batch_idx, len(exp.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
      #             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

      # Save checkpoint.
      acc = 100. * correct / total

      exp.writer.add_scalar('test/acc', acc)
      exp.writer.add_scalar('test/loss', test_loss)

      exp.writer.add_scalar("test/student/eval",student_eval)
      exp.writer.add_scalar("test/teacher/eval", teacher_eval)


  if acc > exp.best_acc:
    print('Saving..')
    state = {
      'net': exp.net.state_dict(),
      'acc': acc,
      'epoch': epoch
    }
    if not os.path.isdir('checkpoint'):
      os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.pth')
    exp.best_acc = acc