import torch.nn as nn


class Flatten(nn.Module):
  def forward(self, x):
    batch_size = x.shape[0]
    return x.view(batch_size, -1)

def linear_model(hidden_size, input_size=3072, out_size=10):
  layers = [input_size] + hidden_size + [out_size]
  mod_lays = []
  mod_lays.append(Flatten())
  for i in range(len(layers) - 2):
    mod_lays.append(nn.Linear(layers[i], layers[i + 1]))
    mod_lays.append(nn.ReLU())
  mod_lays.append(nn.Linear(layers[-2], layers[-1]))

  return nn.Sequential(*mod_lays)