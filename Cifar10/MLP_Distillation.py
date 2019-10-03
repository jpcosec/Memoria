import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from argparse
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
from lib.dist_model import linear_model
from lib.dist_utils import dist_loss_gen, train_op
from lib.utils import sample, get_dataloaders, acc_test
from lib.teacher_models.resnet import ResNet18 as teacherNetGenerator

from torch.utils.tensorboard import SummaryWriter



def dist_model(teacher, distiller, eval_criterion, params, device, loaders,writer):
  train_loader, test_loader = loaders


  for epoch in range(params.epochs):
    for batch_idx, (data, target) in enumerate(train_loader):  # 784= 28*28
      loss = train_op(distiller, teacher, data, target, device)
      #logger.debug(str(loss))


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

      writer.add_scalar('dist-loss/train',loss.detach().numpy())
      writer.add_scalar('CE/train',train_stats)
      writer.add_scalar('CE/test', test_stats)
      writer.add_scalar('Acc/train', acc_test(distiller["model"], train_loader, flatten=True))
      writer.add_scalar('Acc/test', acc_test(distiller["model"], test_loader, flatten=True))



def distillation_experiment(neuronas, teacher, device, loaders, params):

  for i in neuronas:
    for x in range(params.experiments):

      # Lambdear
      stexp=str(i)
      #for neurona in i:
      #  stexp+=str(neurona)+"-"

      stexp+="_exp"+str(x)
      #stexp =str(i) + str(x)#"\r"
      logger.info("experimento " + stexp)

      writer = SummaryWriter(comment=stexp)

      student_model = linear_model([i]).to(device)
      criterion = dist_loss_gen(params.temp)
      optimizer = torch.optim.SGD(student_model.parameters(), lr=0.01)
      eval_criterion = torch.nn.CrossEntropyLoss()

      distiller=dict(model=student_model,criterion=criterion,optimizer=optimizer)

      dist_model(teacher, distiller, eval_criterion, params, device, loaders, writer)

def

def main(params):
  neuronas = [int(i) for i in np.exp2(np.arange(0, 10))]



  torch.set_default_tensor_type('torch.cuda.FloatTensor')
  torch.cuda.current_device()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logger.info('Using device:' + str(device))

  # Get data
  loaders = get_dataloaders(params.data_folder)

  teacher = TeacherNet().to(device)
  logger.info("loading teacher")
  teacher.load_state_dict(torch.load(params.model_path))

  for param in teacher.parameters():
    param.requires_grad = False

  distillation_experiment(neuronas, teacher, device, loaders, params)



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
