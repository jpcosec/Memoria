import torch

import torch.backends.cudnn as cudnn

import os
import json
from lib.utils import get_model
from lib.Experiment import Experiment


class DistillationExperiment(Experiment):  # TODO: solucionar problemas de herencia
  """
  Class created for classification supervised distillation problems
  """

  def __init__(self, **kwargs):
    super(DistillationExperiment, self).__init__(
                  device=kwargs["device"],
                  net=kwargs["student"],
                  optimizer=kwargs["optimizer"],
                  criterion=kwargs["criterion"],
                  linear=kwargs["linear"],
                  writer=kwargs["writer"],
                  testloader=kwargs["testloader"],
                  trainloader=kwargs["trainloader"],
                  best_acc=kwargs["best_acc"]
                )

    self.student = kwargs["student"]
    self.teacher = kwargs["teacher"]
    self.eval_criterion = kwargs["eval_criterion"]

    # variables que se acumulan a lo largo de una epoca para logs
    self.train_dict = {'loss': 0,
                      'total': 0,
                      'correct_student': 0,
                      'correct_teacher': 0,
                      'eval_student':0,
                      "batch_idx":0}

    self.test_dict = {'loss': 0,
                       'total': 0,
                       'correct_student': 0,
                       'correct_teacher': 0,
                       'eval_student':0,
                       "batch_idx":0,
                       }
    # funciones lambda de estadisticos obtenidos sobre esas variables
    self.test_log_funcs = {'student/acc': lambda dict : 100. * dict["correct_student"] / dict["total"],
                          'teacher/acc': lambda dict : 100. * dict["correct_student"] / dict["total"],
                          'loss': lambda dict: dict["loss"] / (dict["batch_idx"] + 1),
                          "student/eval": lambda dict: dict["eval_student"]}

    self.train_log_funcs = {'student/acc': lambda dict : 100. * dict["correct_student"] / dict["total"],
                          'teacher/acc': lambda dict : 100. * dict["correct_student"] / dict["total"],
                          'loss': lambda dict: dict["loss"] / (dict["batch_idx"] + 1),
                          "student/eval": lambda dict: dict["eval_student"]}

    self.include_targets= self.criterion.__name__=="total_loss"

    self.teacher.eval()



  def process_batch(self, inputs, targets, batch_idx):

    if not self.test_phase:
      self.optimizer.zero_grad()

    S_y_pred,predicted = self.net_forward(inputs)
    T_y_pred,predictedT = self.net_forward(inputs, teacher=True)

    loss_dict={"student_scores":S_y_pred, "teacher_scores":T_y_pred}
    if self.include_targets:
      loss_dict["y"]=targets

    loss = self.criterion(**loss_dict)

    self.accumulate_stats(loss=loss.item(),
                          total=targets.size(0),
                          correct_student=predicted.eq(targets).sum().item(),
                          correct_teacher=predictedT.eq(targets).sum().item())

    self.update_stats(batch_idx,eval_student=self.eval_criterion(S_y_pred, targets).item())

    if not self.test_phase:
      loss.backward()
      self.optimizer.step()

    self.record_step()

  def net_forward(self, inputs, teacher=False):
    """
    Method made for hiding the .view choice
    :param inputs:
    :return:
    """
    net=self.teacher if teacher else self.student

    if self.flatten:
      outputs = net(inputs.view(-1, self.flat_dim))
    else:
      outputs = net(inputs)

    _, predicted = outputs.max(1)
    return outputs, predicted



def load_teacher(args, device):
  print('==> Building teacher model..', args.teacher)
  net = get_model(args.teacher)
  net = net.to(device)

  for param in net.parameters():
    param.requires_grad = False

  if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

  assert os.path.isdir(args.teacher), 'Error: model not initialized'
  os.chdir(args.teacher)
  # Load checkpoint.
  print('==> Resuming from checkpoint..')
  assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
  checkpoint = torch.load('./checkpoint/ckpt.pth')
  net.load_state_dict(checkpoint['net'])

  return net


def load_student(args, device):
  best_acc = 0  # best test accuracy
  start_epoch = 0  # start from epoch 0 or last checkpoint epoch
  folder = "student/" + args.student + "/" + args.distillation
  # Model
  print('==> Building student model..', args.student)
  net = get_model(args.student)
  net = net.to(device)
  if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

  if args.resume:
    assert os.path.isdir(folder), 'Error: model not initialized'
    os.chdir(folder)
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['student_acc']
    start_epoch = checkpoint['epoch']

    if start_epoch >= args.epochs:
      print("Number of epochs already trained")
  else:
    if not os.path.isdir("student/"):
      os.mkdir("student")
    if not os.path.isdir("student/" + args.student):
      os.mkdir("student/" + args.student)
    os.mkdir(folder)
    os.chdir(folder)
  return net, best_acc, start_epoch

"""
def train(exp):
  print('\rTraining epoch: %d' % exp.epoch)
  exp.student.train()
  exp.teacher.eval()

  total_loss = 0
  correct = 0
  correctT = 0
  total = 0
  for batch_idx, (inputs, targets) in enumerate(exp.trainloader):  # 784= 28*28
    inputs, targets = inputs.to(exp.device), targets.to(exp.device)
    exp.optimizer.zero_grad()

    # Predecir
    if exp.flatten:  # meter en exp
      S_y_pred = exp.student(inputs.view(-1, 3072))
    else:
      S_y_pred = exp.student(inputs)

    T_y_pred = exp.teacher(inputs)

    # Compute Loss
    # loss = exp.criterion(S_y_pred, T_y_pred)# dejar como *args
    loss = exp.criterion(S_y_pred, T_y_pred, targets)

    # Backward pass
    loss.backward()
    exp.optimizer.step()

    total_loss += loss.item()
    # Accuracy
    total += targets.size(0)
    _, predicted = S_y_pred.max(1)
    correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    # teacher Accuracy
    _, predictedT = T_y_pred.max(1)
    correctT += predictedT.eq(targets).sum().item()
    accT = 100. * correct / total

    loss = total_loss / (batch_idx + 1)

    EC = exp.eval_criterion(S_y_pred, targets).item()

    logs = dict([("loss", loss),
                 ("EC", EC),
                 ("teacher/acc", acc),
                 ("student/acc", accT)])

    exp.record_step(logs)

  exp.record_epoch(logs, acc)


def test(exp):
  print('\rTesting epoch: %d' % exp.epoch)
  exp.student.eval()
  exp.teacher.eval()

  ac_loss = 0  # acumular estos
  student_correct = 0
  teacher_correct = 0
  total = 0

  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(exp.testloader):
      inputs, targets = inputs.to(exp.device), targets.to(exp.device)
      # Predecir
      if exp.flatten:
        S_y_pred = exp.student(inputs.view(-1, 3072))
      else:
        S_y_pred = exp.student(inputs)

      T_y_pred = exp.teacher(inputs)
      # dist_loss = exp.criterion(S_y_pred, T_y_pred)
      dist_loss = exp.criterion(S_y_pred, T_y_pred, targets)

      student_eval = exp.eval_criterion(S_y_pred.squeeze(), targets).item()
      teacher_eval = exp.eval_criterion(T_y_pred.squeeze(), targets).item()

      ac_loss += dist_loss.item()

      _, predicted = S_y_pred.max(1)
      total += targets.size(0)
      student_correct += predicted.eq(targets).sum().item()

      _, predictedT = T_y_pred.max(1)
      teacher_correct += predictedT.eq(targets).sum().item()

      student_acc = 100. * student_correct / total
      teacher_acc = 100. * teacher_correct / total
      loss = ac_loss / total

      # progress_bar(batch_idx, len(exp.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
      #             % (ac_loss / (batch_idx + 1), 100. * student_correct / total, student_correct, total))

      logs = dict([('student/acc', student_acc),
                   ('teacher/acc', teacher_acc),
                   ('loss', loss),
                   ("student/eval", student_eval),
                   ("teacher/eval", teacher_eval)])

      exp.record_step(logs, test=True)

    exp.record_epoch(logs, student_acc, test=True)
"""