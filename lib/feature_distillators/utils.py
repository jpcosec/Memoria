import torch
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict

from lib.kd_distillators.utils import DistillationExperiment


class FeatureExperiment(DistillationExperiment):

    def __init__(self, **kwargs):
        self.teacher_keys = [kwargs["args"].teacher_layer]  # todo: arreglar para multicapa
        self.student_keys = [kwargs["args"].student_layer]

        if "shape" in kwargs:
            self.shape = (kwargs["args"].batch_size, 3, kwargs["shape"], kwargs["shape"])
        else:
            self.shape = (kwargs["args"].batch_size, 3, 32, 32)

        self.kd_criterion = kwargs["kd_criterion"]

        self.ft_criterion = kwargs["ft_criterion"]
        self.criterion = self.kd_criterion

        if "alpha" in kwargs:
            self.alpha = kwargs["alpha"]
        else:
            self.alpha = 1.0

        # If self.use_features or self.feature_train:
        self.idxs = kwargs['idxs']

        super(FeatureExperiment, self).__init__(**kwargs, criterion=self.kd_criterion)

        self.feature_train = True
        self.kd_train = True

        self.__create_feature_extraction()

        if "use_regressor" in kwargs.keys():
            self.use_regressor = kwargs["use_regressor"]
        else:
            self.use_regressor = True

        if self.use_regressor:
            self.__create_regressors()

    def __create_feature_extraction(self):
        self.teacher_features = {}
        self.teacher_layers = 1

        def register_teacher_hook(module):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            m_key = "%s-%i" % (class_name, self.teacher_layers)

            def hook(mod, inp, out):
                self.teacher_features[m_key] = out

            if (
                    not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
                    and not (module == self.teacher)
            ):
                # print(m_key)
                if self.teacher_layers in self.teacher_keys:
                    module.register_forward_hook(hook)
                self.teacher_layers += 1

        self.teacher.apply(register_teacher_hook)

        self.student_features = {}
        self.student_layers = 1

        def register_student_hook(module):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            m_key = "%s-%i" % (class_name, self.student_layers)

            def hook(mod, inp, out):
                self.student_features[m_key] = out

            if (
                    not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
                    and not (module == self.student)
            ):
                # print(m_key)
                if self.student_layers in self.student_keys:
                    module.register_forward_hook(hook)
                self.student_layers += 1

        self.student.apply(register_student_hook)

        inp = torch.rand(*self.shape).to(self.device)

        self.teacher.eval()
        self.student.eval()

        _ = self.teacher(inp)
        _ = self.student(inp)

        s_sizes = [tensor.shape for tensor in list(self.student_features.values())]
        t_sizes = [tensor.shape for tensor in list(self.teacher_features.values())]

        for s, t in zip(s_sizes, t_sizes):
            if s[-1] != t[-1] or s[-2] != t[-2]:
                raise AttributeError("size mismatch")

    def __create_regressors(self):
        inp = torch.rand(1,*self.shape[2:]).to(self.device)
        self.teacher.eval()
        self.student.eval()
        out = self.teacher(inp)
        out2 = self.student(inp)

        s_sizes = [tensor.shape for tensor in list(self.student_features.values())]
        t_sizes = [tensor.shape for tensor in list(self.teacher_features.values())]

        self.regressors = [torch.nn.Conv2d(s_sizes[i][1],
                                           t_sizes[i][1],
                                           kernel_size=1
                                           ).to(self.device) for i in range(len(self.idxs))]

        self.regressor_optimizers = [optim.Adam(r.parameters(), lr=0.001) for r in self.regressors]

    def process_batch(self, inputs, targets, batch_idx):  # todo: Cambiar  y loss a dict

        s_output, predicted = self.net_forward(inputs)
        t_output, predictedt = self.net_forward(inputs, teacher=True)

        loss = torch.tensor(0.0, requires_grad=True).to(self.device)  # todo: meter alphas
        # todo: meter loss en applt loss
        if self.feature_train:
            sf = list(self.student_features.values())[0]  # todo: arreglar para caso multicapa
            tf = list(self.teacher_features.values())

            if self.use_regressor:
                sf = self.regressors[0](sf)  # self.regressors[0](sf[0])
            floss = self.ft_criterion(tf[0], sf)  # self.alpha*self.ft_criterion(tf[0], sf)
            loss += floss
            # todo: Cambiar esta wea a iterable

        if self.kd_train:
            loss_dict = {"input": s_output, "teacher_logits": t_output, "target": targets, }
            kd_loss = self.kd_criterion(**dict([(field, loss_dict[field]) for field in self.criterion_fields]))

            loss += kd_loss

            self.accumulate_stats(correct_student=predicted.eq(targets).sum().item(),
                                  correct_teacher=predictedt.eq(targets).sum().item(),
                                  eval_student=self.eval_criterion(s_output, targets).item())

        self.accumulate_stats(loss=loss.item(),
                              total=targets.size(0))
        self.update_stats(batch_idx)

        if not self.test_phase:

            self.optimizer.zero_grad()

            if self.use_regressor:
                for o in self.regressor_optimizers:
                    o.zero_grad()

            loss.backward(retain_graph=True)

            if self.use_regressor:
                for o in self.regressor_optimizers:
                    o.step()

            self.optimizer.step()  # todo: ver que pasa cuando se hace pre-train

        self.record_step()
