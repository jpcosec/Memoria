import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from argparse import ArgumentParser
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
from lib.dist_model import linear_model
from lib.dist_utils import dist_loss_gen
from lib.utils import get_dataloaders, test
from lib.teacher_model import Net as TeacherNet

import torch.multiprocessing as mp

def train(distiller,teacher,loaders,device):
    # Construct data_loader, optimizer, etc.
    train_loader, test_loader = loaders

    for data, labels in train_loader:
        x_train = data.to(device)
        y_train = labels.to(device)
        logger.debug("lasorr")

        distiller["optimizer"].zero_grad()
        # Forward pass

        # Predecir
        S_y_pred = distiller["student_model"](x_train.view(-1, 784))
        T_y_pred = teacher(x_train)

        # Compute Loss
        loss = distiller["criterion"](S_y_pred, T_y_pred)
        # Backward pass
        loss.backward()
        distiller["optimizer"].step()


def initialize(params):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.current_device()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device:' + str(device))

    # Get data
    loaders = get_dataloaders(params.data_folder)

    teacher = TeacherNet().to(device)
    logger.info("loading teacher")
    teacher.load_state_dict(torch.load(params.model_path))
    teacher.eval()

    return loaders, teacher, device

def create_distiller(neuronas,params,device):
    # one student, returns dict without teacher


    student_model = linear_model(neuronas).to(device)
    student_model.train()
    criterion = dist_loss_gen(params.temp)
    optimizer = torch.optim.SGD(student_model.parameters(), lr=0.01)
    eval_criterion = torch.nn.CrossEntropyLoss()

    return dict(criterion=criterion, student_model=student_model,optimizer=optimizer,eval_criterion=eval_criterion)


def main(params):
    loaders, teacher, device = initialize(params)
    distiller=create_distiller([100], params, device)

    num_processes = 4

    # NOTE: this is required for the ``fork`` method to work
    teacher.share_memory()
    distiller.student_model.share_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(distiller,teacher,loaders,device,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

if __name__ == '__main__':
    # Arg_parsing
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