
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable



def train(net,train_loader,optimizer,device,n_epochs=100):


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






