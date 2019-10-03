import lib.teacher_models

def get_model(model_name):
  model_list = dict(VGG=lib.teacher_models.vgg.VGG('VGG19'),
                        ResNet18=lib.teacher_models.resnet.ResNet18(),
                        ResNet50=lib.teacher_models.resnet.ResNet50(),
                        ResNet101=lib.teacher_models.resnet.ResNet101(),
                        MobileNet=lib.teacher_models.mobilenet.MobileNet(),
                        MobileNetV2=lib.teacher_models.mobilenetv2.MobileNetV2(),
                        ResNeXt29=lib.teacher_models.resnext.ResNeXt29_32x4d(),
                        DenseNet=lib.teacher_models.densenet.DenseNet121(),
                        PreActResNet18=lib.teacher_models.preact_resnet.PreActResNet18(),
                        DPN92=lib.teacher_models.dpn.DPN92(),
                        SENet18=lib.teacher_models.senetSENet18(),
                        EfficientNetB0=lib.teacher_models.efficientnet.EfficientNetB0(),
                        GoogLeNet=lib.teacher_models.googlenet.GoogLeNet(),
                        ShuffleNetG2=lib.teacher_models.shufflenet.ShuffleNetG2(),
                        ShuffleNetV2=lib.teacher_models.shufflenetv2.ShuffleNetV2(1))
  try:
    return model_list[model_name]
  except:
    raise ModuleNotFoundError("Model not found")
