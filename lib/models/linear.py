import torch.nn as nn


def linear_model(hidden_size, input_size=3072, out_size=10):
  layers = [input_size] + hidden_size + [out_size]
  mod_lays = []
  for i in range(len(layers) - 2):
    mod_lays.append(nn.Linear(layers[i], layers[i + 1]))
    mod_lays.append(nn.ReLU())
    nn.Dropout(0.5)
  mod_lays.append(nn.Linear(layers[-2], layers[-1]))
  nn.Dropout(0.5)

  return nn.Sequential(*mod_lays)