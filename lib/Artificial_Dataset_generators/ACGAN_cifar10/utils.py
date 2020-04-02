import numpy as np
from matplotlib import pyplot as plt
import torch

def showImage(images,epoch=-99):

    images = images.cpu().numpy()
    images = images/2 + 0.5
    plt.imshow(np.transpose(images,axes = (1,2,0)))
    plt.axis('off')
    if epoch!=-99:
        plt.savefig("outs/e" + str(epoch) + ".png")



def save_samples(tensor, start=0,folder="samples",batch_size=128):
    from PIL import Image
    from numpy import split, squeeze
    #images = images.cpu().numpy()

    tensor =tensor / 2 + 0.5


    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    #print(tensor.shape)
    ndarr = tensor.mul_(127.5).add_(0.5).clamp_(0, 255).permute(0,2,3,1).to('cpu', torch.uint8).numpy()
    #print(ndarr.shape)
    for n,arr in enumerate(split(ndarr,batch_size)):
        #print(squeeze(arr).shape)
        im = Image.fromarray(squeeze(arr)).resize((32,32))
        im.save(folder+"/"+str(n+start)+".png")
    return n+start


    #plt.imshow(np.transpose(images, axes=(1, 2, 0)))

    #plt.axis('off')
    #if epoch != -99:
    #    plt.savefig("outs/e" + str(epoch) + ".png")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



