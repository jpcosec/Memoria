from lib.models.densenet import DenseNet121
from lib.models.dpn import DPN92
from lib.models.efficientnet import EfficientNetB0
from lib.models.googlenet import GoogLeNet
from lib.models.linear import linear_model
from lib.models.mobilenet import MobileNet
from lib.models.mobilenetv2 import MobileNetV2
from lib.models.preact_resnet import PreActResNet18
from lib.models.resnet import ResNet18, ResNet50, ResNet101
from lib.models.resnext import ResNeXt29_32x4d
from lib.models.senet import SENet18
from lib.models.vgg import VGG
from lib.models.mnist_net import MnistNet
from lib.utils.data.cifar10 import load_samples, additive_noise, load_cifar10
from lib.utils.funcs import auto_change_dir

""" Model Stuff"""

def get_model(model_name):

    if model_name.split("_")[0] == "linear":
        auto_change_dir("linear")
        shape = [int(st) for st in model_name.split("_")[1].split(",")]
        return linear_model(shape)

    model_list = dict(VGG=VGG('VGG19'),
                      ResNet18=ResNet18(),
                      ResNet50=ResNet50(),
                      ResNet101=ResNet101(),
                      MobileNet=MobileNet(),
                      MobileNetV2=MobileNetV2(),
                      ResNeXt29=ResNeXt29_32x4d(),
                      DenseNet=DenseNet121(),
                      PreActResNet18=PreActResNet18(),
                      DPN92=DPN92(),
                      SENet18=SENet18(),
                      EfficientNetB0=EfficientNetB0(),
                      GoogLeNet=GoogLeNet(),
                      MnistNet=MnistNet()
                      )
    try:
        return model_list[model_name]
    except:
        raise ModuleNotFoundError("Model not found")


def cifar10_parser(args):

    print("usando dataset", args.dataset)

    if args.dataset=="vae_sample":
        return load_samples(args, "VAE_SAMP")

    transform, arg= args.transform.split(",")
    print("usando transformacion", transform, "con args",arg)
    if transform=="none":
        transform_train= None
    else:
        t_dict={
            "noise": additive_noise
        }

        transform_train = t_dict[transform](float(arg))

    return load_cifar10(args, transform_train=transform_train)