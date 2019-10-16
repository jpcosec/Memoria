from lib.utils.Experiment import Experiment
from numpy import random

class dummy_experiment(Experiment):
    """
    Class created for dummy testing
    """

    def __init__(self, **kwargs):
        exp = Experiment(device='cpu',
                         net=None,
                         optimizer=None,
                         criterion=None,
                         linear=None,
                         writer=writer,
                         testloader=None,
                         trainloader=None,
                         best_acc=None,
                         start_epoch=start_epoch
                         )

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


    def process_batch(self, inputs, targets, batch_idx):

        if not self.test_phase:
            self.optimizer.zero_grad()

        S_y_pred, predicted = self.net_forward(inputs)
        T_y_pred, predictedT = self.net_forward(inputs, teacher=True)

        loss_dict = {"student_scores": S_y_pred, "teacher_scores": T_y_pred}
        if self.include_targets:
            loss_dict["y"] = targets

        loss = self.criterion(**loss_dict)

        self.accumulate_stats(loss=loss.item(),
                              total=targets.size(0),
                              correct_student=predicted.eq(targets).sum().item(),
                              correct_teacher=predictedT.eq(targets).sum().item())

        self.update_stats(batch_idx, eval_student=self.eval_criterion(S_y_pred, targets).item())

        if not self.test_phase:
            loss.backward()
            self.optimizer.step()

        self.record_step()

    def net_forward(self):
        """
        Method made for hiding the .view choice
        :param teacher:
        :param inputs:
        :return:
        """
        outputs=random.rand(10)

        _, predicted = outputs.max(1)
        return outputs, predicted