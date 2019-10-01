import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from argparse import ArgumentParser
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
from lib.dist_model import linear_model
from lib.dist_utils import dist_loss_gen, train_op
from lib.utils import sample, get_dataloaders, test
from lib.teacher_model import Net as TeacherNet


def dist_model(teacher, distiller, epochs,  eval_criterion, params, device, loaders):
  train_loader, test_loader = loaders


  history = {"epoch": [],
             "train": [],
             "test": [],
             "loss": [],
             "acc_train": [],
             "acc_test": []
             }


  for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):  # 784= 28*28
      loss = train_op(distiller,teacher,data,target,device)
      logger.debug(str(loss))


    if epoch % 1 == 0:
      x_train, y_train = sample(train_loader)

      y_pred = distiller["model"](x_train.view(-1, 784))
      train_stats = eval_criterion(y_pred, y_train)

      x_test, y_test = sample(test_loader)
      y_pred = distiller["model"](x_test.view(-1, 784))
      test_stats = eval_criterion(y_pred.squeeze(), y_test)

      y_predT = teacher(x_test)
      test_statsT = eval_criterion(y_predT.squeeze(), y_test)

      # todo JSONEAR O ALGO
      history["epoch"].append(epoch)
      history["loss"].append(loss.item())
      history["test"].append(test_stats.item())
      history["train"].append(train_stats.item())
      history["acc_test"].append(test(distiller["model"], train_loader, flatten=True))
      history["acc_train"].append(test(distiller["model"], test_loader, flatten=True))
      # print('Epoch {}: train loss: {}, test loss: {}'.format(epoch, loss.item(), test_stats.item()) )

  return history


def distillation_experiment(neuronas, epochs, temp, teacher, device, loaders, params):
  exps = {}
  dist_models = {}

  for i in neuronas:
    trains = []
    tests = []
    losses = []
    acc_train = []
    acc_test = []

    for x in range(params.experiments):
      print("\r", i, x, end='')
      logger.debug("experimento ")

      student_model = linear_model([i]).to(device)
      criterion = dist_loss_gen(temp)
      optimizer = torch.optim.SGD(student_model.parameters(), lr=0.01)
      eval_criterion = torch.nn.CrossEntropyLoss()
      logger.debug("problemon")


      history = dist_model(teacher, student_model, epochs, criterion, eval_criterion, optimizer, params,device, loaders)

      trains.append(history["train"])
      tests.append(history["test"])
      losses.append(history["loss"])
      acc_train.append(history["acc_train"])
      acc_test.append(history["acc_test"])

    exps[i] = {"epoch": history["epoch"],
               "train": np.array(trains).mean(axis=0),
               "test": np.array(tests).mean(axis=0),
               "loss": np.array(losses).mean(axis=0),
               "acc_train": np.array(acc_train).mean(axis=0),
               "acc_test": np.array(acc_test).mean(axis=0)
               }
    # models[i] = student_model

  # plot_exp(exps)
  return exps


def main(params):
  neuronas = [int(i) for i in np.exp2(np.arange(0, 10))]


  torch.set_default_tensor_type('torch.cuda.FloatTensor')
  torch.cuda.current_device()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logger.info('Using device:' + str(device))



  # Get data
  loaders= get_dataloaders(params.data_folder)

  teacher = TeacherNet().to(device)
  logger.info("loading teacher")
  teacher.load_state_dict(torch.load(params.model_path))

  for param in teacher.parameters():
    param.requires_grad = False

  ex = distillation_experiment(neuronas, params.epochs, params.temp, teacher, device, loaders, params)

  p = pd.DataFrame.from_dict(ex)

  with open("expDist%f.csv" % params.temp, "w") as text_file:
    text_file.write(p.to_csv(index=True))


if __name__ == '__main__':


  parser = ArgumentParser()
  parser.add_argument("--save_model", type=bool, default=True)
  parser.add_argument("--train_model", type=bool, default=False)
  parser.add_argument("--model_path", type=str, default="mnist_cnn.pt")
  parser.add_argument("--data_folder", type=str, default="./data")
  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument("--temp", type=float, default=3.5)
  parser.add_argument("--experiments", type=int, default=2)

  hparams = parser.parse_args()

  main(hparams)
