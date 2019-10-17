import torch
import torch.backends.cudnn as cudnn

from lib.utils.Experiment import Experiment
from lib.utils.utils import get_model
import os


class HintExperiment(Experiment):

  def __init__(self, **kwargs):
    super(HintExperiment, self).__init__(**kwargs, net=kwargs["student"])

    self.student = kwargs["student"]
    self.student_features = kwargs["student_features"]
    self.teacher = kwargs["teacher"]
    self.teacher_features = kwargs["teacher_features"]

    self.regressors=kwargs["regressors"]
    self.regressor_optimizers=kwargs["regressor_optim"]

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
    self.test_log_funcs = {  # 'acc': lambda dict: 100. * dict["correct_student"] / dict["total"],
      # 'teacher/acc': lambda dict: 100. * dict["correct_student"] / dict["total"],
      'loss': lambda dict: dict["loss"] / (dict["batch_idx"] + 1),
      # "eval": lambda dict: dict["eval_student"]
    }

    self.train_log_funcs = {  # 'acc': lambda dict: 100. * dict["correct_student"] / dict["total"],
      # 'teacher/acc': lambda dict: 100. * dict["correct_student"] / dict["total"],
      'loss': lambda dict: dict["loss"] / (dict["batch_idx"] + 1),
      # "eval": lambda dict: dict["eval_student"]
    }

    self.teacher.eval()
    self.student.train()
    # self.criterion_fields = self.criterion.__code__.co_varnames

  def process_batch(self, inputs, targets, batch_idx):

    if not self.test_phase:
      self.optimizer.zero_grad()
      for o in self.regressor_optimizers:
        o.zero_grad()

    S_y_pred, predicted = self.net_forward(inputs)
    T_y_pred, predictedT = self.net_forward(inputs, teacher=True)

    loss = self.criterion(self.teacher_features[0],self.student_features[0],self.regressors[0])#todo: Cambiar esta wea a iterable

    # loss_dict = {"input": S_y_pred, "teacher_logits": T_y_pred, "target": targets,}

    # loss = self.criterion(**dict([(field, loss_dict[field]) for field in self.criterion_fields]))  # probar

    self.accumulate_stats(loss=loss.item(),
                          total=targets.size(0),
                          #correct_student=predicted.eq(targets).sum().item(),
                          #correct_teacher=predictedT.eq(targets).sum().item()
                          )

    self.update_stats(batch_idx)#, eval_student=self.eval_criterion(S_y_pred, targets).item())

    if not self.test_phase:
      print("LASORRA")
      loss.backward()
      self.optimizer.step()
      for o in self.regressor_optimizers:
        o.step()

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
