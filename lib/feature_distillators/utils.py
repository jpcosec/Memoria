import torch

from lib.kd_distillators.utils import DistillationExperiment


class HintExperiment(DistillationExperiment):

    def __init__(self, **kwargs):
        super(HintExperiment, self).__init__(**kwargs, criterion=None)

        self.student = kwargs["student"]
        self.student_features = kwargs["student_features"]
        self.teacher = kwargs["teacher"]
        self.teacher_features = kwargs["teacher_features"]

        self.kd_criterion = kwargs["kd_criterion"]
        self.ft_criterion = kwargs["ft_criterion"]

        self.regressors = kwargs["regressors"]
        self.regressor_optimizers = kwargs["regressor_optim"]

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
        self.test_log_funcs = {  # "acc": lambda dict : 1,
            'acc': lambda di: 100. * di["correct_student"] / di["total"],
            'teacher/acc': lambda di: 100. * di["correct_student"] / di["total"],
            'loss': lambda di: di["loss"] / (di["batch_idx"] + 1),
            "eval": lambda di: di["eval_student"]
        }

        self.train_log_funcs = {  # 'acc': lambda dict : 1,
            "acc": lambda di: 100. * di["correct_student"] / di["total"],
            'teacher/acc': lambda di: 100. * di["correct_student"] / di["total"],
            'loss': lambda di: di["loss"] / (di["batch_idx"] + 1),
            "eval": lambda di: di["eval_student"]
        }

        self.teacher.eval()
        self.student.train()
        self.criterion_fields = self.kd_criterion.__code__.co_varnames

        self.feature_train = True
        self.kd_train = True

    def process_batch(self, inputs, targets, batch_idx):

        if not self.test_phase:
            self.optimizer.zero_grad()
            for o in self.regressor_optimizers:
                o.zero_grad()

        s_output, predicted = self.net_forward(inputs)
        t_output, predictedt = self.net_forward(inputs, teacher=True)

        loss = torch.tensor(0.0, requires_grad=True).to(self.device)  # todo: meter alphas
        # todo: meter loss en applt loss
        if self.feature_train:
            r = self.regressors[0](self.student_features[0])
            floss = self.ft_criterion(self.teacher_features[0], r)
            loss += floss
            # todo: Cambiar esta wea a iterable y a if

        if self.kd_train:
            loss_dict = {"input": s_output, "teacher_logits": t_output, "target": targets, }
            kd_loss = self.kd_criterion(**dict([(field, loss_dict[field]) for field in self.criterion_fields]))
            loss += kd_loss

            self.accumulate_stats(loss=loss.item(),
                                  total=targets.size(0),
                                  correct_student=predicted.eq(targets).sum().item(),
                                  correct_teacher=predictedt.eq(targets).sum().item()
                                  )

        self.update_stats(batch_idx, eval_student=self.eval_criterion(s_output, targets).item())

        if not self.test_phase:
            loss.backward(retain_graph=True)
            if self.kd_train:
                self.optimizer.step()
            if self.feature_train:
                for o in self.regressor_optimizers:
                    o.step()

        self.record_step()
