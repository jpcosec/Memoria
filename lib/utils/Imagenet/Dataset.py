import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.dataset import random_split, Dataset
import pandas as pd

torch.manual_seed(0)


def get_imageNet(class_file="lib/utils/Imagenet/Imagenet_classes",
                 image_folder="/home/jpruiz/PycharmProjects/ImageNet-Datasets-Downloader/imagenet_images"):



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
        classname = path.split("/")[-2]
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
