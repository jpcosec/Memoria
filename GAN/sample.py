from GAN.model import Generator
import torch

from PIL import Image
from numpy import split, squeeze


def save_samples(tensor, start=0, folder="samples", batch_size=128):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    #print(tensor.shape)
    ndarr = tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(0,2, 3, 1).to('cpu', torch.uint8).numpy()
    for n,arr in enumerate(split(ndarr, batch_size)):
        #print(squeeze(arr).shape)
        im = Image.fromarray(squeeze(arr))
        im.save(folder+"/"+str(n+start)+".png")
    return n+start


from torchvision.utils import save_image
# checking the availability of cuda devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# number of gpu's available
ngpu = 1
# input noise dimension
nz = 100
# number of generator filters
ngf = 64

weights_folder = "./EXP2/weights/"
epoch = 56

n_samples = 10000

# Initialization

model = Generator(ngpu).to(device)

path = weights_folder+'netG_epoch_%i.pth'%epoch

print('==> Resuming from checkpoint..', path)


model.load_state_dict(torch.load(path))
model.eval()
#load(checkpoint)

def main():

    for i in range(1, n_samples // 128):
        noise=torch.randn(128, nz, 1, 1, device=device)
        fake = model(noise)

        save_samples(fake, start=i*128)


if __name__ == "__main__":
    main()