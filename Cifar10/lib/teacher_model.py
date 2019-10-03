import .teacher_models as models

def get_model(model_name):
  teacher_models = dict(VGG=models.vgg.VGG('VGG19'),
                        ResNet18=models.resnet.ResNet18(),
                        ResNet50=models.resnet.ResNet50(),
                        ResNet101=models.resnet.ResNet101(),
                        MobileNet=models.mobilenet.MobileNet(),
                        MobileNetV2=models.mobilenetv2.MobileNetV2(),
                        ResNeXt29=models.resnext.ResNeXt29_32x4d(),
                        DenseNet=models.densenet.DenseNet121(),
                        PreActResNet18=models.preact_resnet.PreActResNet18(),
                        DPN92=models.dpn.DPN92(),
                        SENet18=models.senetSENet18(),
                        EfficientNetB0=models.efficientnet.EfficientNetB0(),
                        GoogLeNet=models.googlenet.GoogLeNet(),
                        ShuffleNetG2=models.shufflenet.ShuffleNetG2(),
                        ShuffleNetV2=models.shufflenetv2.ShuffleNetV2(1))
  try:
    return teacher_models[model_name]
  except:
    raise ModuleNotFoundError("Model not found")
