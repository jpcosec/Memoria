import argparse
import os

import torch
import torch.nn as nn
from torchvision.utils import make_grid
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image

from lib.Artificial_Dataset_generators.ACGAN_cifar10.model import Generator, Discriminator
from lib.Artificial_Dataset_generators.ACGAN_cifar10.utils import showImage, weights_init

from lib.utils.funcs import auto_change_dir, check_folders

os.chdir("../../../Cifar10")
print(os.getcwd())




parser = argparse.ArgumentParser(description='GAN Dataset sampler')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=400, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=25, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--sample', action='store_true', default=False,
                    help='samples trained model')
parser.add_argument('--folder',  default="GAN-Dataset",
                    help='output folder')
args = parser.parse_args()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


tf = transforms.Compose([transforms.Resize(64),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])

trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True,
                                     transform = tf)



trainloader = torch.utils.data.DataLoader(trainset, batch_size = 128, shuffle = True)


#print(len(dataset))
#print(dataset[0][0].size())
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'fake')

auto_change_dir(args.folder)
check_folders()


dataiter = iter(trainloader)
images,labels = dataiter.next()
print(images.size())
showImage(make_grid(images[0:64]))

# custom weights initialization called on netG and netD

gen = Generator().to(device)
gen.apply(weights_init)

disc = Discriminator().to(device)
disc.apply(weights_init)

paramsG = list(gen.parameters())
print(len(paramsG))

paramsD = list(disc.parameters())
print(len(paramsD))        
        
optimG = optim.Adam(gen.parameters(), 0.0002, betas = (0.5,0.999))
optimD = optim.Adam(disc.parameters(), 0.0002, betas = (0.5,0.999))

validity_loss = nn.BCELoss()

real_labels = 0.7 + 0.5 * torch.rand(10, device = device)
fake_labels = 0.3 * torch.rand(10, device = device)


for epoch in range(1,args.epochs+1):
    
    for idx, (images,labels) in enumerate(trainloader):
        
        batch_size = images.size(0)

        labels= labels.to(device)
        images = images.to(device)
        
        real_label = real_labels[idx % 10]
        fake_label = fake_labels[idx % 10]
        
        fake_class_labels = 10*torch.ones((batch_size,),dtype = torch.long,device = device)
        
        if idx % args.log_interval == 0:
            real_label, fake_label = fake_label, real_label
        
        # ---------------------
        #         disc
        # ---------------------
        
        optimD.zero_grad()       
        
        # real
        validity_label = torch.full((batch_size,),real_label , device = device)
   
        pvalidity, plabels = disc(images)       
        
        errD_real_val = validity_loss(pvalidity, validity_label)            
        errD_real_label = F.nll_loss(plabels,labels)
        
        errD_real = errD_real_val + errD_real_label
        errD_real.backward()
        
        D_x = pvalidity.mean().item()        
        
        #fake 
        noise = torch.randn(batch_size,100,device = device)
        sample_labels = torch.randint(0,10,(batch_size,),device = device, dtype = torch.long)
        
        fakes = gen(noise,sample_labels)
        
        validity_label.fill_(fake_label)
        
        pvalidity, plabels = disc(fakes.detach())       
        
        errD_fake_val = validity_loss(pvalidity, validity_label)
        errD_fake_label = F.nll_loss(plabels, fake_class_labels)
        
        errD_fake = errD_fake_val + errD_fake_label
        errD_fake.backward()
        
        D_G_z1 = pvalidity.mean().item()
        
        #finally update the params!
        errD = errD_real + errD_fake
        
        optimD.step()
    
        
        # ------------------------
        #      gen
        # ------------------------
        
        
        optimG.zero_grad()
        
        noise = torch.randn(batch_size,100,device = device)  
        sample_labels = torch.randint(0,10,(batch_size,),device = device, dtype = torch.long)
        
        validity_label.fill_(1)
        
        fakes = gen(noise,sample_labels)
        pvalidity,plabels = disc(fakes)
        
        errG_val = validity_loss(pvalidity, validity_label)        
        errG_label = F.nll_loss(plabels, sample_labels)
        
        errG = errG_val + errG_label
        errG.backward()
        
        D_G_z2 = pvalidity.mean().item()
        
        optimG.step()
        if idx%args.log_interval==0:
        
            print("[{}/{}] [{}/{}] D_x: [{:.4f}] D_G: [{:.4f}/{:.4f}] G_loss: [{:.4f}] D_loss: [{:.4f}] D_label: [{:.4f}] "
                  .format(epoch,args.epochs, idx, len(trainloader),D_x, D_G_z1,D_G_z2,errG,errD,
                          errD_real_label + errD_fake_label + errG_label))

    with torch.no_grad():
        noise = torch.randn(100,100,device = device)
        labels = torch.arange(0,100,dtype = torch.long,device = device)//10
        images = gen(noise, labels)
        showImage(images,epoch)

    
    torch.save(gen.state_dict(),'checkpoints/gen_%i.pth'%epoch)
    torch.save(disc.state_dict(),'checkpoints/disc%i.pth'%epoch)
