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


    # Instantiate net
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    if train_model:
        train_net(net,train_loader)
    else:
        net.load_state_dict(torch.load(PATH))

    net.eval()

    print("net_accuracy", test(net, test_loader))