# Library of wrappers for model distillation

import lib.models

class ResNetDistillator(lib.models.resnet.ResNet):
  def __init__(self, block, num_blocks, num_classes=10):
    super(ResNetDistillator,self).__init__(block,num_blocks,num_classes=num_classes)

  def forward_with_features(self,x, returns=[1,2,3,4,5]):
    outs=[]
    outs.append(F.relu(self.bn1(self.conv1(x))))
    outs.append(self.layer1(outs[-1]))
    outs.append(self.layer2(outs[-1]))
    outs.append(self.layer3(outs[-1]))
    outs.append(self.layer4(outs[-1]))
    outs.append(F.avg_pool2d(outs[-1], 4))
    outs.append(outs[-1].view(outs[-1].size(0), -1))
    outs.append(self.linear(outs[-1]))
    return [o for i,o in enumerate(outs) if i in returns]


# hacer uno para sequentials