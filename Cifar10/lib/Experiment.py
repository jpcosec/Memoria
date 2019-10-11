import json
import os

import torch


class Experiment():#todo mover a clase independiente
  """
  Objecto for regular supervised tasks
   
  """#todo documentar
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

    try:
      with open('record.json', 'w') as fp:
        self.record=json.load(fp)
        self.epoch=self.record["epoch"]
        self.train_step = self.record["train_step"]
        self.test_step = self.record["test_step"]

    except:
      self.record={}
      self.epoch = 0
      self.train_step = 0
      self.test_step = 0

  def record_step(self, logs, test=False):#todo: meter variable bool test/train
    """
    Saves logs to tb.writer and advances one step
    :param logs:
    :param test:
    :return:
    """
    if test:
      for field,value in logs.items():
        self.writer.add_scalar("test/"+field, value, global_step=self.test_step)
      self.test_step += 1
    else:
      for field,value in logs.items():
        self.writer.add_scalar("train/"+field, value, global_step=self.train_step)
      self.train_step += 1

  def record_epoch(self, logs, acc, test=False):
    """
    Saves logs to json and advances one epoch
    :param logs:
    :param acc:
    :param test:
    :return:
    """
    phase="test" if test else "train"
    print("\rEpoch %i %s stats\n"%(self.epoch,phase),logs)

    self.record.update({self.epoch:{phase:logs}})

    if test:
      self.save_model(acc)
      self.epoch+=1

  def save_model(self,acc):
    # Early stoping, # Save checkpoint.
    if acc > self.best_acc:
      print('Saving..')
      state = {
        'net': self.net.state_dict(),
        'student_acc': acc,
        'epoch': self.epoch
      }

      if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
      torch.save(state, './checkpoint/ckpt.pth')
      self.best_acc = acc

      self.record.update({"epoch":self.epoch})
      self.record.update({"train_step": self.train_step})
      self.record.update({"test_step": self.test_step})

      with open('record.json', 'w') as fp:
        json.dump(self.record, fp)

  def net_prediction(self,**args):#todo: llenar abstracta
    #
    pass

  def batch_process(self):#todo: llenar abstracta
    pass

  def train_epoch(self):#todo: llenar
    # Se inicializan variables de acumulacion

    # Se itera sobre
    pass


  def test_epoch(self):#todo: llenar
    pass