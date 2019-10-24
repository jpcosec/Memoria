import json
import os

import torch


class Experiment:
    """
    Objecto for regular supervised tasks
     
    """  # todo documentar

    def __init__(self, **kwargs):
        self.device = kwargs["device"]
        self.net = kwargs["net"]
        self.optimizer = kwargs["optimizer"]
        self.criterion = kwargs["criterion"]
        self.flatten = kwargs["linear"]
        self.writer = kwargs["writer"]
        self.testloader = kwargs["testloader"]
        self.trainloader = kwargs["trainloader"]
        self.best_acc = kwargs["best_acc"]


        if "dimensions" in kwargs:
            self.flat_dim = kwargs["dimensions"]
        else:
            self.flat_dim = 3072

        print("flat dimensions of", self.flat_dim)

        self.load_record()
        if "args" in kwargs:
            d=vars(kwargs["args"])
            with open('config.json', 'w') as fp:
                json.dump(d, fp)


        self.last_acc = 1.0

        self.test_phase = True

        # variables que se acumulan a lo largo de una epoca para logs
        self.test_dict = {'loss': 0,
                          'total': 0,
                          'correct': 0,
                          "batch_idx": 0}

        self.train_dict = {'loss': 0,
                           'total': 0,
                           'correct': 0,
                           "batch_idx": 0}

        # funciones lambda de estadisticos obtenidos sobre esas variables
        self.train_log_funcs = {'acc': lambda stats_dict: 100. * stats_dict["correct"] / stats_dict["total"],
                                'loss ': lambda stats_dict: stats_dict["loss"] / (stats_dict["batch_idx"] + 1)
                                }
        self.test_log_funcs = {'acc': lambda stats_dict: 100. * stats_dict["correct"] / stats_dict["total"],
                               'loss ': lambda stats_dict: stats_dict["loss"] / (stats_dict["batch_idx"] + 1)
                               }

    def load_record(self):
        try:
            with open('record.json', 'r') as fp:
                self.record = json.load(fp)
                self.epoch = self.record["epoch"]
                self.train_step = self.record["train_step"]
                self.test_step = self.record["test_step"]

        except:
            self.record = {"test": {}, "train": {}}
            self.epoch = 0
            self.train_step = 0
            self.test_step = 0

    def record_step(self):  # todo: meter variable bool test/train
        """
        Saves logs to tb.writer and advances one step
        :param logs:
        :param test:
        :return:
        """

        stats_dict = self.test_dict if self.test_phase else self.train_dict
        func_dict = self.test_log_funcs if self.test_phase else self.train_log_funcs
        logs = dict([(k, func(stats_dict)) for k, func in func_dict.items()])

        if self.test_phase:
            for field, value in logs.items():
                self.writer.add_scalar("test/" + field, value, global_step=self.test_step)

            self.test_step += 1
        else:
            for field, value in logs.items():
                self.writer.add_scalar("train/" + field, value, global_step=self.train_step)
            self.train_step += 1
        #print(logs)#todo borrar

    def record_epoch(self):
        stats_dict = self.test_dict if self.test_phase else self.train_dict
        func_dict = self.test_log_funcs if self.test_phase else self.train_log_funcs
        logs = dict([(k, func(stats_dict)) for k, func in func_dict.items()])

        if self.test_phase:
            self.last_acc = logs["acc"]

        phase = "test" if self.test_phase else "train"
        print("\rEpoch %i %s stats\n" % (self.epoch, phase), logs, )

        self.record[phase]=({self.epoch: logs})#devolver a estado normal despues

        self.save_model()

    def accumulate_stats(self, **arg_dict):
        stats_dict = self.test_dict if self.test_phase else self.train_dict
        for key, value in arg_dict.items():
            stats_dict[key] += value

    def update_stats(self, batch_idx, **arg_dict):
        stats_dict = self.test_dict if self.test_phase else self.train_dict
        stats_dict["batch_idx"] = batch_idx
        for key, value in arg_dict.items():
            stats_dict[key] = value

    def save_model(self, save_checkpoints=True, overwrite_record=True):
        # Early stoping, # Save checkpoint.

        if self.last_acc > self.best_acc:
            print('Saving..')
            state = {
                'net': self.net.state_dict(),
                'acc': self.last_acc,
                'epoch': self.epoch
            }
            self.best_acc = self.last_acc
            self.record.update(dict([("epoch", self.epoch),
                                     ("train_step",self.train_step),
                                     ("test_step", self.test_step)]))
            with open('record.json', 'w') as fp:
                json.dump(self.record, fp)

            if save_checkpoints:
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/ckpt.pth')


        if overwrite_record:
            with open('record.json', 'w') as fp:
                json.dump(self.record, fp)

    def train_epoch(self):
        self.__set_train_phase()
        print('\rTraining epoch: %d' % self.epoch, )
        self.iterate_epoch(self.trainloader, self.train_dict)

    def test_epoch(self):
        self.__set_test_phase()
        print('\rTesting epoch: %d' % self.epoch, )
        with torch.no_grad():
            self.iterate_epoch(self.testloader, self.test_dict)
        self.epoch += 1

    def __set_train_phase(self):
        self.net.train()
        self.test_phase = False

    def __set_test_phase(self):
        self.net.eval()
        self.test_phase = True

    def iterate_epoch(self, loader, stats_dict):

        # Se inicializan variables de acumulacion
        for k in stats_dict.keys():
            stats_dict[k] = 0

        # se itera sobre dataset
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.process_batch(inputs, targets, batch_idx)

        self.record_epoch()

    def process_batch(self, inputs, targets, batch_idx):
        outputs = self.net_forward(inputs)
        loss = self.criterion(outputs, targets)

        _, predicted = outputs.max(1)

        if not self.test_phase:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.accumulate_stats(loss=loss.item(),
                              total=targets.size(0),
                              correct=predicted.eq(targets).sum().item())
        self.update_stats(batch_idx)

        self.record_step()

    def net_forward(self, inputs):
        """
        Method made for hiding the .view choice
        :param inputs:
        :return:
        """
        if self.flatten:
            outputs = self.net(inputs.view(-1, self.flat_dim))
        else:
            outputs = self.net(inputs)
        return outputs
