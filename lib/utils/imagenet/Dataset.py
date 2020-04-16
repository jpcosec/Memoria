import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.dataset import random_split, Dataset
import pandas as pd

from lib.utils.funcs import auto_change_dir

torch.manual_seed(0)


def get_imageNet(class_file="C:/Users/PC/PycharmProjects/Memoria/lib/utils/Imagenet/Imagenet_classes",
                 #class_file="/home/jpruiz/PycharmProjects/Memoria/lib/utils/imagenet/Imagenet_classes",

                 image_folder="C:/Users/PC/PycharmProjects/ImageNet-Datasets-Downloader/dataset/imagenet_images"):
                 #image_folder="/home/jpruiz/PycharmProjects/ImageNet-datasets-downloader/dataset/imagenet_images"):

    classes = [i.split(":")[-1] for i in open(class_file).read().replace("}", "").split('\n')]
    classes = [i.replace("'", '') for i in classes]
    serie = pd.Series(classes)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    def get_I2012_class(path):
        # global serie
        #classname = path.split("/")[-2]
        classname = path.split("\\")[-2]
        # print("class",classname,path)
        return serie.index[serie.str.contains(classname)][0]

    class Dataset_final(Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.subset)

    dataset = datasets.ImageFolder(image_folder)
    dataset.samples = [(path, get_I2012_class(path)) for (path, i) in dataset.samples]
    l1 = int(len(dataset) * 0.8)

    lengths = [l1, int(len(dataset) - l1)]
    trainset, testset = random_split(dataset, lengths)
    trainset = Dataset_final(trainset, transform=train_transform)
    testset = Dataset_final(testset, transform=test_transform)
    return trainset, testset


def get_dataloaders(batch_size, folder=None):
    trainset, testset = get_imageNet()

    auto_change_dir("Imagenet")
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size, shuffle=False, num_workers=0)
    return trainloader, testloader
