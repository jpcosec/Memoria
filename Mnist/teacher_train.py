import torch

import torch.utils.data.dataloader as dataloader
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST

from lib.teacher_model import Net
from lib.teacher_utils import train, test



model_path = "mnist_cnn.pt"
data_folder = './data'
train_model = True




torch.set_default_tensor_type('torch.cuda.FloatTensor')

torch.cuda.current_device()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Load MNIST

train_data = MNIST(data_folder, train=True, download=True, transform=transforms.Compose([
  transforms.ToTensor(),  # ToTensor does min-max normalization.
]), )

test_data = MNIST(data_folder, train=False, download=True, transform=transforms.Compose([
  transforms.ToTensor(),  # ToTensor does min-max normalization.
]), )

# Create DataLoader
dataloader_args = dict(shuffle=True, batch_size=64, num_workers=1, pin_memory=True)
train_loader = dataloader.DataLoader(train_data, **dataloader_args)
test_loader = dataloader.DataLoader(test_data, **dataloader_args)

# Instantiate net
net = Net().to(device)

if train_model:
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  train(net, train_loader, optimizer, device)
  save_model = True
  if (save_model):
    torch.save(net.state_dict(), model_path)

else:
  net.load_state_dict(torch.load(model_path))

net.eval()

print("net_accuracy", test(net, test_loader))
