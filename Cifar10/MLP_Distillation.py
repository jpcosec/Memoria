import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from argparse
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter


from lib.models.linear import linear_model
from lib.student.utils import dist_loss_gen, train, test
from lib.utils import get_dataloaders


def distillation_experiment(neuronas, teacher, device, loaders, params):
  train_loader, test_loader = loaders
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

      for epoch in range(params.epochs):
        train(distiller, teacher, train_loader, device, writer)
        test(distiller, teacher, loaders, device, eval_criterion, writer)



def main(params):
  neuronas = [int(i) for i in np.exp2(np.arange(0, 10))]



  torch.set_default_tensor_type('torch.cuda.FloatTensor')
  torch.cuda.current_device()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logger.info('Using device:' + str(device))

  # Get data
  loaders = get_dataloaders(params.data_folder)





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
