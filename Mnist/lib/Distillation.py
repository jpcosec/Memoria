import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim

from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import MNIST


def dist_loss_gen(T=8):
  def dist_loss(student_scores, teacher_scores, T=T):
    return nn.KLDivLoss()(F.log_softmax(student_scores / T, dim=1), F.softmax(teacher_scores / T, dim=1))

  return dist_loss


def sample(loader):
  data, target = next(iter(loader))
  data, target = Variable(data.cuda()), Variable(target.cuda())
  return data, target


def dist_model(T_model, S_model, epochs, criterion, eval_criterion, optimizer):
  global train_loader, test_loader
  # train_loader, x_test,y_test = experiment_data()
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

      history["epoch"].append(epoch)
      history["loss"].append(loss.item())
      history["test"].append(test_stats.item())
      history["train"].append(train_stats.item())
      history["acc_test"].append(test(S_model, train_loader))
      history["acc_train"].append(test(S_model, test_loader))
      # print('Epoch {}: train loss: {}, test loss: {}'.format(epoch, loss.item(), test_stats.item()) )

  return history


def distillation_experiment(neuronas, epochs, temp, teacher, experiments=2):
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

      history = dist_model(teacher, student_model, epochs, criterion, eval_criterion, optimizer)
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

  plot_exp(exps)
  return exps


def linear_model(hidden_size, input_size=784, out_size=10):
  layers = [input_size] + hidden_size + [out_size]
  mod_lays = []
  for i in range(len(layers) - 2):
    mod_lays.append(nn.Linear(layers[i], layers[i + 1]))
    mod_lays.append(torch.nn.ReLU())
  mod_lays.append(nn.Linear(layers[-2], layers[-1]))

  return nn.Sequential(*mod_lays)

if __name__ == '__main__':

  PATH = "mnist_cnn.pt"
  data_folder = './data'
  train_model = False

  torch.set_default_tensor_type('torch.cuda.FloatTensor')

  torch.cuda.current_device()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('Using device:', device)

  # Load MNIST

  train = MNIST(data_folder, train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),  # ToTensor does min-max normalization.
  ]), )

  test = MNIST(data_folder, train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),  # ToTensor does min-max normalization.
  ]), )

  # Create DataLoader
  dataloader_args = dict(shuffle=True, batch_size=64, num_workers=1, pin_memory=True)
  train_loader = dataloader.DataLoader(train, **dataloader_args)
  test_loader = dataloader.DataLoader(test, **dataloader_args)

  neuronas = [int(i) for i in np.exp2(np.arange(0, 10))]

  epochs = 50
  temp = 3.5

  teacher = net
  for param in teacher.parameters():
    param.requires_grad = False

  ex = distillation_experiment(neuronas, epochs, temp, teacher, experiments=1)

  p = pd.DataFrame.from_dict(ex)

  with open("expDist%f.csv" % temp, "w") as text_file:
    text_file.write(p.to_csv(index=True))