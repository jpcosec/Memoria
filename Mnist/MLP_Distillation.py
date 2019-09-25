from argparse import ArgumentParser
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
from lib.dist_model import linear_model
from lib.dist_utils import dist_loss_gen
from lib.utils import sample, get_dataloaders, test
from lib.teacher_model import Net as TeacherNet


def dist_model(T_model, S_model, epochs, criterion, eval_criterion, optimizer,params):
  train_loader, test_loader = get_dataloaders(params.data_folder)


  history = {"epoch": [],
             "train": [],
             "test": [],
             "loss": [],
             "acc_train": [],
             "acc_test": []
             }

  S_model.train()
  T_model.eval()

  for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):  # 784= 28*28
      x_train, y_train = Variable(data.cuda()), Variable(target.cuda())
      optimizer.zero_grad()
      # Forward pass

      # Predecir
      S_y_pred = S_model(x_train.view(-1, 784))
      T_y_pred = T_model(x_train)

      # Compute Loss
      loss = criterion(S_y_pred, T_y_pred)
      # Backward pass
      loss.backward()
      optimizer.step()

    if epoch % 1 == 0:
      x_train, y_train = sample(train_loader)

      y_pred = S_model(x_train.view(-1, 784))
      train_stats = eval_criterion(y_pred, y_train)

      x_test, y_test = sample(test_loader)
      y_pred = S_model(x_test.view(-1, 784))
      test_stats = eval_criterion(y_pred.squeeze(), y_test)
      y_predT = T_model(x_test)
      test_statsT = eval_criterion(y_predT.squeeze(), y_test)

      # todo JSONEAR O ALGO
      history["epoch"].append(epoch)
      history["loss"].append(loss.item())
      history["test"].append(test_stats.item())
      history["train"].append(train_stats.item())
      history["acc_test"].append(test(S_model, train_loader))
      history["acc_train"].append(test(S_model, test_loader))
      # print('Epoch {}: train loss: {}, test loss: {}'.format(epoch, loss.item(), test_stats.item()) )

  return history


def distillation_experiment(neuronas, epochs, temp, teacher, device,experiments=2):
  exps = {}
  dist_models = {}

  for i in neuronas:
    trains = []
    tests = []
    losses = []
    acc_train = []
    acc_test = []

    for x in range(experiments):
      print("\r", i, x, end='')
      student_model = linear_model([i]).to(device)
      criterion = dist_loss_gen(temp)
      optimizer = torch.optim.SGD(student_model.parameters(), lr=0.01)
      eval_criterion = torch.nn.CrossEntropyLoss()

      history = dist_model(teacher, student_model, epochs, criterion, eval_criterion, optimizer,params)
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


def main(params, neuronas):
  torch.set_default_tensor_type('torch.cuda.FloatTensor')

  torch.cuda.current_device()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('Using device:', device)

  # Get data
  train_loader, test_loader = get_dataloaders(params.data_folder)

  teacher = TeacherNet().to(device)
  print("loading teacher")
  teacher.load_state_dict(torch.load(params.model_path))

  for param in teacher.parameters():
    param.requires_grad = False

  ex = distillation_experiment(neuronas, params.epochs, params.temp, teacher,device, experiments=1)

  p = pd.DataFrame.from_dict(ex)

  with open("expDist%f.csv" % params.temp, "w") as text_file:
    text_file.write(p.to_csv(index=True))


if __name__ == '__main__':
  neuronas = [int(i) for i in np.exp2(np.arange(0, 10))]

  parser = ArgumentParser()
  parser.add_argument("--save_model", type=bool, default=True)
  parser.add_argument("--train_model", type=bool, default=False)
  parser.add_argument("--model_path", type=str, default="mnist_cnn.pt")
  parser.add_argument("--data_folder", type=str, default="./data")
  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument("--temp", type=float, default=3.5)

  hparams = parser.parse_args()

  main(hparams, neuronas)
