import os

import torch
import torch.backends.cudnn as cudnn

from lib.utils.Experiment import Experiment
from lib.utils.utils import get_model


class DistillationExperiment(Experiment):
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
                       'eval_student': 0,
                       "batch_idx": 0}

    self.test_dict = {'loss': 0,
                      'total': 0,
                      'correct_student': 0,
                      'correct_teacher': 0,
                      'eval_student': 0,
                      "batch_idx": 0,
                      }

    # funciones lambda de estadisticos obtenidos sobre esas variables
    self.test_log_funcs = {'acc': lambda dict: 100. * dict["correct_student"] / dict["total"],
                           'teacher/acc': lambda dict: 100. * dict["correct_student"] / dict["total"],
                           'loss': lambda dict: dict["loss"] / (dict["batch_idx"] + 1),
                           "eval": lambda dict: dict["eval_student"]}

    self.train_log_funcs = {'acc': lambda dict: 100. * dict["correct_student"] / dict["total"],
                            'teacher/acc': lambda dict: 100. * dict["correct_student"] / dict["total"],
                            'loss': lambda dict: dict["loss"] / (dict["batch_idx"] + 1),
                            "eval": lambda dict: dict["eval_student"]}

    self.include_targets = self.criterion.__name__ == "total_loss"

    self.teacher.eval()

    self.criterion_fields = self.criterion.__code__.co_varnames()

  def process_batch(self, inputs, targets, batch_idx):

    if not self.test_phase:
      self.optimizer.zero_grad()

    S_y_pred, predicted = self.net_forward(inputs)
    T_y_pred, predictedT = self.net_forward(inputs, teacher=True)

    loss_dict = {"student_scores": S_y_pred, "teacher_scores": T_y_pred, "targets": targets}

    loss = self.criterion(dict([(field, loss_dict[field]) for field in self.criterion_fields]))  # probar

    self.accumulate_stats(loss=loss.item(),
                          total=targets.size(0),
                          correct_student=predicted.eq(targets).sum().item(),
                          correct_teacher=predictedT.eq(targets).sum().item())

    self.update_stats(batch_idx, eval_student=self.eval_criterion(S_y_pred, targets).item())

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
    net = self.teacher if teacher else self.student

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
  folder = "students/" + args.student + "/" + args.distillation
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
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    if start_epoch >= args.epochs:
      print("Number of epochs already trained")

  else:
    if not os.path.isdir("students/"):
      os.mkdir("students")
    if not os.path.isdir("students/" + args.student):
      os.mkdir("students/" + args.student)
    os.mkdir(folder)
    os.chdir(folder)
  return net, best_acc, start_epoch
