
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data.dataloader as dataloader
from torchvision import transforms
from torchvision.datasets import MNIST



def teacher_train(net, train_loader, optimizer, device, n_epochs=100):


    net.train()
    losses = []
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Get Samples
            data=data.to(device)
            target=target.to(device)
            # Init
            optimizer.zero_grad()

            # Predict
            y_pred = net(data)

            # Calculate loss
            loss = F.cross_entropy(y_pred, target)
            losses.append(loss)
            # Backpropagation
            loss.backward()
            optimizer.step()


            # Display
            if batch_idx % 100 == 1:
                print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),

                    100. * batch_idx / len(train_loader),
                    loss),
                    end='')




def test(net, loader, lim=10):
  d = []
  for batch_idx, (data, target) in enumerate(loader):
    data, target = Variable(data.cuda()), Variable(target.cuda())

    output = net(data)
    pred = output.data.max(1)[1]
    d.extend(pred.eq(target).cpu())
    if batch_idx > lim:
      continue

  d = np.array(d)
  accuracy = float(np.sum(d)) / float(d.shape[0])
  return accuracy



def get_dataloaders(data_folder):
  # Load MNIST

  train_data = MNIST(data_folder, train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),  # ToTensor does min-max normalization.
  ]), )

  test_data = MNIST(data_folder, train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),  # ToTensor does min-max normalization.
  ]), )

  # Create DataLoader
  dataloader_args = dict(shuffle=True, batch_size=64, num_workers=0)
  train_loader = dataloader.DataLoader(train_data, **dataloader_args)
  test_loader = dataloader.DataLoader(test_data, **dataloader_args)


  return train_loader, test_loader


def sample(loader):
  data, target = next(iter(loader))
  data, target = Variable(data.cuda()), Variable(target.cuda())
  return data, target