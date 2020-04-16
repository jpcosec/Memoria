import torch
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict

from lib.kd_distillators.utils import DistillationExperiment
import logging

logging.basicConfig(filename='inspection.log', level=logging.DEBUG)


class FeatureExperiment(DistillationExperiment):

    def __init__(self, **kwargs):
        self.teacher_keys = [kwargs["args"].teacher_layer]  # todo: arreglar para multicapa
        self.student_keys = [kwargs["args"].student_layer]

        self.kd_criterion = kwargs["kd_criterion"]

        self.ft_criterion = kwargs["ft_criterion"]
        self.criterion = self.kd_criterion

        if "alpha" in kwargs:
            self.alpha = kwargs["alpha"]
        else:
            self.alpha = 1.0

        if "shape" in kwargs:
            self.shape = kwargs['shape']
        else:
            self.shape = 32

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
        self.student_features = {}

        def get_modules(model, batch_size=-1):
            device = self.device
            input_size = (3, self.shape, self.shape)

            def register_hook(module):

                def hook(module, input, output):
                    class_name = str(module.__class__).split(".")[-1].split("'")[0]
                    module_idx = len(summary)

                    m_key = module_idx + 1  # "%s-%i" % (class_name, module_idx + 1)
                    summary[m_key] = OrderedDict()
                    summary[m_key]["input_shape"] = list(input[0].size())
                    summary[m_key]["input_shape"][0] = batch_size

                    summary[m_key]["module"] = module
                    if isinstance(output, (list, tuple)):
                        summary[m_key]["output_shape"] = [
                            [-1] + list(o.size())[1:] for o in output
                        ]
                    else:
                        summary[m_key]["output_shape"] = list(output.size())
                        summary[m_key]["output_shape"][0] = batch_size

                    params = 0
                    if hasattr(module, "weight") and hasattr(module.weight, "size"):
                        params += torch.prod(torch.LongTensor(list(module.weight.size())))
                        summary[m_key]["trainable"] = module.weight.requires_grad
                    if hasattr(module, "bias") and hasattr(module.bias, "size"):
                        params += torch.prod(torch.LongTensor(list(module.bias.size())))
                    summary[m_key]["nb_params"] = params

                if (
                        not isinstance(module, nn.Sequential)
                        and not isinstance(module, nn.ModuleList)
                        and not (module == model)
                ):
                    hooks.append(module.register_forward_hook(hook))

            device = device.lower()
            assert device in [
                "cuda",
                "cpu",
            ], "Input device is not valid, please specify 'cuda' or 'cpu'"

            if device == "cuda" and torch.cuda.is_available():
                dtype = torch.cuda.FloatTensor
            else:
                dtype = torch.FloatTensor

            # multiple inputs to the network
            if isinstance(input_size, tuple):
                input_size = [input_size]

            # batch_size of 2 for batchnorm
            x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
            # print(type(x[0]))

            # create properties
            summary = OrderedDict()
            hooks = []

            # register hook
            model.apply(register_hook)

            # make a forward pass
            # print(x.shape)
            model(*x)

            # remove these hooks
            for h in hooks:
                h.remove()

            return summary

        logging.info("Hooking teacher features %s" % str(self.teacher_keys))
        teacher_modules = get_modules(self.teacher)

        logging.info("Hooking student features %s" % str(self.student_keys))
        student_modules = get_modules(self.student)

        for key in self.teacher_keys:
            def hook(module, input, output):
                self.teacher_features[key] = output
                pass

            teacher_modules[key]['module'].register_forward_hook(hook)

        for key in self.student_keys:
            def hook(module, input, output):
                self.student_features[key] = output
                pass

            student_modules[key]['module'].register_forward_hook(hook)

    def __create_regressors(self):
        inp = torch.rand(2, 3, self.shape, self.shape).to(self.device)
        self.teacher.eval()
        self.student.eval()
        out = self.teacher(inp)
        out2 = self.student(inp)

        s_sizes = [tensor.shape for tensor in list(self.student_features.values()) if tensor is not None]
        t_sizes = [tensor.shape for tensor in list(self.teacher_features.values()) if tensor is not None]

        self.regressors = [torch.nn.Conv2d(s_sizes[i][1],
                                           t_sizes[i][1],
                                           kernel_size=1
                                           ).to(self.device) for i in range(len(s_sizes))]

        self.regressor_optimizers = [optim.Adam(r.parameters(), lr=0.001) for r in self.regressors]

    def process_batch(self, inputs, targets, batch_idx):  # todo: Cambiar  y loss a dict

        s_output, predicted = self.net_forward(inputs)
        t_output, predictedt = self.net_forward(inputs, teacher=True)

        loss = torch.tensor(0.0, requires_grad=True).to(self.device)  # todo: meter alphas
        # todo: meter loss en applt loss
        if self.feature_train:  # todo: arreglar para caso multicapa
            Fs = [tensor for tensor in list(self.student_features.values()) if tensor is not None][0]
            Ft = [tensor for tensor in list(self.teacher_features.values()) if tensor is not None][0]

            # Fs = list(self.student_features.values())[0]
            # Ft = list(self.teacher_features.values())

            if self.use_regressor:
                Fs = self.regressors[0](Fs)  # self.regressors[0](sf[0])

            floss = self.ft_criterion(Ft, Fs)  # self.alpha*self.ft_criterion(tf[0], sf)# incorporar al tb
            logging.debug(str(floss))
            logging.debug(str(Ft[0].mean()))
            logging.debug(str(Fs[0].mean()))
            # print(floss)
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
