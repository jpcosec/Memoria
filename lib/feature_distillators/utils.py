import torch
import torch.optim as optim

from lib.kd_distillators.utils import DistillationExperiment
from lib.utils.utils import register_hooks
from lib.models.resnet import ResNet as resnet





class HintExperiment(DistillationExperiment):

    def __init__(self, **kwargs):
        super(HintExperiment, self).__init__(**kwargs, criterion=None)



        self.kd_criterion = kwargs["kd_criterion"]
        self.ft_criterion = kwargs["ft_criterion"]

        #self.regressors = kwargs["regressors"]
        #self.regressor_optimizers = kwargs["regressor_optim"]

        self.criterion_fields = self.kd_criterion.__code__.co_varnames

        self.feature_train = True
        self.kd_train = True
        self.f_lambda = 0.000001


        idxs=[4]

        self.teacher_features = []

        for name, module in self.teacher._modules.items():
            for id, layer in enumerate(module.children()):
                if id in idxs:
                    def hook(m, i, o):
                        self.teacher_features.append(o)
                    layer.register_forward_hook(hook)

        self.student_features = []
        for name, module in self.student._modules.items():
            for id, layer in enumerate(module.children()):
                if id in idxs:
                    def hook(m, i, o):
                        self.student_features.append(o)
                    layer.register_forward_hook(hook)

        #student_features = [f[1] for f in fs.items()]
        #teacher_features = [f[1] for f in ft.items()]
        inp = torch.rand(1, 3, 32, 32).to(self.device)
        self.teacher.eval()
        self.student.eval()
        out = self.teacher()
        out2 = self.student()

        self.regressors = [torch.nn.Conv2d(self.student_features[i].shape[1],
                                      self.teacher_features[i].shape[1],
                                      kernel_size=1).to(self.device)
                                      for i in range(len(idxs))]

        self.regressor_optimizers = [optim.Adam(r.parameters(), lr=0.001) for r in self.regressors]


        def register_hooks(net, idxs, feature):
            """
            Registers a hook in the module of the net
            :param net:
            :param idxs:
            :param feature:
            :return:
            """

            def hook(m, i, o):
                feature[m] = o

            for name, module in net._modules.items():
                for id, layer in enumerate(module.children()):
                    if id in idxs:
                        layer.register_forward_hook(hook)


        #self.

    def process_batch(self, inputs, targets, batch_idx):

        s_output, predicted = self.net_forward(inputs)
        t_output, predictedt = self.net_forward(inputs, teacher=True)

        loss = torch.tensor(0.0, requires_grad=True).to(self.device)  # todo: meter alphas
        # todo: meter loss en applt loss
        if self.feature_train:
            #hooked_layers = [4]
            #fs = {}
            #ft = {}
            #register_hooks(self.teacher, hooked_layers, ft)
            #register_hooks(self.student, hooked_layers, fs)
            #student_features = [f[1] for f in fs.items()]
            #teacher_features = [f[1] for f in ft.items()]

            r = self.regressors[0](self.student_features[0])
            floss = self.f_lambda*self.ft_criterion(self.teacher_features[0], r)
            loss += floss
            # todo: Cambiar esta wea a iterable

        if self.kd_train:
            loss_dict = {"input": s_output, "teacher_logits": t_output, "target": targets, }
            kd_loss = self.kd_criterion(**dict([(field, loss_dict[field]) for field in self.criterion_fields]))

            loss += kd_loss

            self.accumulate_stats(correct_student=predicted.eq(targets).sum().item(),
                                  correct_teacher=predictedt.eq(targets).sum().item(),
                                  eval_student=self.eval_criterion(s_output, targets).item())



        #print(floss/kd_loss)
        self.accumulate_stats(loss=loss.item(),
                              total=targets.size(0))
        #print(loss)
        self.update_stats(batch_idx)

        if not self.test_phase and False:
            self.optimizer.zero_grad()
            for o in self.regressor_optimizers:
                o.zero_grad()

            loss.backward(retain_graph=True)

            if self.feature_train:
                for o in self.regressor_optimizers:
                   o.step()

            if self.kd_train:
                self.optimizer.step()

        self.record_step()
