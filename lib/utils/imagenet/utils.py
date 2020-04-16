import torch.nn as nn
import torchvision.models as models
from lib.utils.funcs import auto_change_dir


def load_model(model_name, trainable=True, device="cuda"):
    model_dict = {
        "ResNet18": models.resnet18,
        "ResNet152": models.resnet152,
        "MobileNetV2": models.mobilenet_v2
    }

    model = model_dict[model_name]
    # print(model)
    auto_change_dir(model_name)

    if trainable:
        model = model(pretrained=False)
    else:
        model = model(pretrained=True, progress=True)
        for param in model.parameters():
            param.requires_grad = False

    model = model.to(device)
    if device == 'cuda':
        model = nn.DataParallel(model)

    return model


def get_blocks(modelname, block):
    d = {"ResNet18": [3, 17, 33, 49, 65],
         "ResNet152": [3, 39, 121, 483, 511],
         "MobileNetV2": [12, 30, 57, 120, 156]}
    return d[modelname][block]
