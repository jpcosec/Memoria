import os

import torch
import torch.backends.cudnn as cudnn

from lib.utils.Experiment import Experiment
from lib.utils.utils import get_model
from lib.utils.funcs import auto_change_dir


class DistillationExperiment(Experiment):
  """
     Class created for classification supervised distillation problems
     """
  def __init__(self, **kwargs):
    super(DistillationExperiment, self).__init__(**kwargs, net=kwargs["student"])

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
                           'teacher/acc': lambda dict: 100. * dict["correct_teacher"] / dict["total"],
                           'loss': lambda dict: dict["loss"] / (dict["batch_idx"] + 1),
                           "eval": lambda dict: dict["eval_student"]/ (dict["batch_idx"] + 1)}

    self.train_log_funcs = {'acc': lambda dict: 100. * dict["correct_student"] / dict["total"],
                            'teacher/acc': lambda dict: 100. * dict["correct_teacher"] / dict["total"],
                            'loss': lambda dict: dict["loss"] / (dict["batch_idx"] + 1),
                            "eval": lambda dict: dict["eval_student"]/ (dict["batch_idx"] + 1)}

    self.teacher.eval()
    self.last_test_acc=01.0

    self.criterion_fields = self.criterion.__code__.co_varnames


  def process_batch(self, inputs, targets, batch_idx):

    S_y_pred, predicted = self.net_forward(inputs)
    T_y_pred, predictedT = self.net_forward(inputs, teacher=True)

    loss_dict = {"input": S_y_pred, "teacher_logits": T_y_pred, "target": targets}

    loss = self.criterion(**dict([(field, loss_dict[field]) for field in self.criterion_fields
                                  if field in loss_dict.keys()]))  #todo: ver si se puede harcodear

    self.accumulate_stats(loss=loss.item(),
                          total=targets.size(0),
                          correct_student=predicted.eq(targets).sum().item(),
                          correct_teacher=predictedT.eq(targets).sum().item(),
                          eval_student=self.eval_criterion(S_y_pred, targets).item())

    self.update_stats(batch_idx)

    if not self.test_phase:
      self.optimizer.zero_grad()
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

  auto_change_dir(args.teacher)
  print(os.getcwd())
  # Load checkpoint.
  print('==> Resuming from checkpoint..')
  assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
  checkpoint = torch.load('./checkpoint/ckpt.pth')
  net.load_state_dict(checkpoint['net'])

  return net

def silent_load(model, device):
    print('==> Building model..', model)
    net = get_model(model)
    net = net.to(device)

    for param in net.parameters():
      param.requires_grad = False

    if device == 'cuda':
      net = torch.nn.DataParallel(net)
      cudnn.benchmark = True
    return net

def load_student(args, device,base_folder="students"):

  auto_change_dir("/".join([base_folder,args.student]))

  print('==> Building student model..', args.student)
  net = get_model(args.student)
  net = net.to(device)

  if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

  try:#if args.resume and os.path.isdir('checkpoint'):#Cambiar resume a otra wea
    # Load checkpoint.
    print('==> Resuming from checkpoint..')

    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    if start_epoch >= args.epochs:
      print("Number of epochs already trained")

  except:#else:
    print('==> Brand new beginning..')
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

  return net, best_acc, start_epoch
