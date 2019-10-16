from argparse import ArgumentParser

import torch
import torch.optim as optim
from lib.teacher_model import Net

from lib.utils.utils import teacher_train, acc_test, load_mnist


def main(params):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.current_device()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Get data
    train_loader, test_loader = load_mnist(params.data_folder)

    # Instantiate net
    net = Net().to(device)

    if params.train_model:
        print("training teacher from scrathc")
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        teacher_train(net, train_loader, optimizer, device)
        if (params.save_model):
            torch.save(net.state_dict(), params.model_path)

    else:
        print("loading teacher")
        net.load_state_dict(torch.load(params.model_path))

    net.eval()

    print("net_accuracy", acc_test(net, test_loader))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--train_model", type=bool, default=False)
    parser.add_argument("--model_path", type=str, default="mnist_cnn.pt")
    parser.add_argument("--data_folder", type=str, default="./data")

    hparams = parser.parse_args()

    main(hparams)
