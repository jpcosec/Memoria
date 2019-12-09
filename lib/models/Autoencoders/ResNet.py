"""
Refactored from https://github.com/arnaghosh/Auto-Encoder/blob/master/resnet.py

"""


import os
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.optim as optim
from torchvision import models, transforms

from lib.models.Autoencoders.ResNet_utils import *

zsize = 48
batch_size = 11
iterations = 500
learningRate = 0.0001






encoder = Encoder(Bottleneck, [3, 4, 6, 3])
encoder.load_state_dict(torch.load(
  '/home/deepkliv/Downloads/resnet50-19c8e357.pth'))  # ,map_location=lambda storage, loc: storage.cuda(1)),strict=False)
# loaded_weights = torch.load('/home/siplab/Saket/resnet18-5c106cde.pth')
# print encoder.layer1[1].conv1.weight.data[0][0]
encoder.fc = nn.Linear(2048, 48)
# for param in encoder.parameters():
#    param.requires_grad = False
encoder = encoder.cuda()
y = torch.rand(1, 3, 224, 224)
x = torch.rand(1, 128)
x = Variable(x.cuda())


# print decoder(x)
# y=Variable(y.cuda())
# print("\n")
# encoder(y)
# print encoder(y)
##########################################################################


binary = Binary()


##########################################################################


decoder = Decoder()


##########################################


# print Autoencoder()

autoencoder = Autoencoder()


# autoencoder = torch.nn.DataParallel(autoencoder, device_ids=[0, 1, 2])


# print Classifier()
classifier = Classifier()


# classifier = torch.nn.DataParallel(classifier, device_ids=[0, 1, 2])


# print Classification()
classification = Classification()

##########################

if torch.cuda.is_available():
  autoencoder.cuda()
  classification.cuda()
  decoder.cuda()
  encoder.cuda()
  classifier.cuda()
# data

plt.ion()

use_gpu = torch.cuda.is_available()
if use_gpu:
  pinMem = True  # Flag for pinning GPU memory
  print('GPU is available!')
else:
  pinMem = False
net = models.resnet18(pretrained=False)
transform = transforms.Compose(
  [
    transforms.Scale((224, 224), interpolation=2),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
  ])
trainset = torchvision.datasets.ImageFolder("/home/deepkliv/Desktop/AE/ram/AE_classifier/fashion/dataset/train",
                                            transform=transform, target_transform=None)
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=batch_size, num_workers=2)
testset = torchvision.datasets.ImageFolder("/home/deepkliv/Desktop/AE/ram/AE_classifier/fashion/dataset/test",
                                           transform=transform, target_transform=None)
testloader = torch.utils.data.DataLoader(testset, shuffle=True, batch_size=batch_size, num_workers=2)

autoencoder_criterion = nn.MSELoss()
classification_criterion = nn.NLLLoss()

autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=learningRate)
classification_optimizer = optim.Adam(classification.parameters(), lr=learningRate)
# encoder_optimizer = optim.Adam(Encoder.parameters(), lr = learningRate)
list_a_loss = []
list_c_loss = []

# fig = plt.figure()
for epoch in range(iterations):
  run_loss = 0
  run_c_loss = 0
  autoencoder.train(True)  # For training
  classification.train(True)
  for i, data in enumerate(trainloader):
    # print i
    inputs, labels = data
    inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

    autoencoder_optimizer.zero_grad()
    classification_optimizer.zero_grad()
    # print(inputs.size())
    pred = autoencoder(inputs)
    # torchvision.utils.save_image(pred.data[0:8], os.path.join('/home/deepkliv/Saket/AE_Classifier/', 'batch_%d_%d'%((epoch+1)/1,i+1) + '.jpg'))
    a_loss = autoencoder_criterion(pred, inputs)
    a_loss.backward()
    autoencoder_optimizer.step()

    # print("efc3", autoencoder.encoder.fc3.bias.grad)

    class_pred = classification(inputs)

    c_loss = classification_criterion(class_pred, labels)

    # _,xxpred = torch.max(class_pred.data, 1)
    # print("class_pred")
    # print(xxpred.cpu().numpy())
    c_loss.backward(retain_graph=True)
    classification_optimizer.step()
    # encoder_optimizer.step()

    run_loss += a_loss.data[0]
    run_c_loss += c_loss.data[0]
    # print i
    if (i + 1) % 2 == 0:
      print(
        '[%d, %5d] Autoencoder loss: %.3f Classification loss: %.3f' % (epoch + 1, i + 1, run_loss / 2, run_c_loss / 2))
      # print('[%d,%5d] Classification loss: %.3f' % (epoch + 1, i + 1, run_c_loss/10))
      run_c_loss = 0.0
      run_loss = 0.0

    decoder_path = os.path.join('/home/deepkliv/Desktop/AE/ram/AE_classifier/fashion/Decoder/',
                                'decoder-%d.pkl' % (epoch + 1))
    encoder_path = os.path.join('/home/deepkliv/Desktop/AE/ram/AE_classifier/fashion/Encoder/',
                                'encoder-%d.pkl' % (epoch + 1))
    autoencoder_path = os.path.join('/home/deepkliv/Desktop/AE/ram/AE_classifier/fashion/Autoencoder/',
                                    'autoencoder-%d.pkl' % (epoch + 1))
    classifier_path = os.path.join('/home/deepkliv/Desktop/AE/ram/AE_classifier/fashion/Classifier/',
                                   'classifier-%d.pkl' % (epoch + 1))
    classification_path = os.path.join('/home/deepkliv/Desktop/AE/ram/AE_classifier/fashion/Classification',
                                       'classification-%d.pkl' % (epoch + 1))

    torch.save(decoder.state_dict(), decoder_path)
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(autoencoder.state_dict(), autoencoder_path)
    torch.save(classifier.state_dict(), classifier_path)
    torch.save(classification.state_dict(), classification_path)

  if (epoch + 1) % 1 == 0:
    list_a_loss.append(run_loss / 5000)
    list_c_loss.append(run_c_loss / 5000)

    # plt.plot(range(epoch+1),list_a_loss,'r--',label='autoencoder')
    # plt.plot(range(epoch+1),list_c_loss,'b--',label='classifier')
    # if epoch==0:
    # plt.legend(loc='upper left')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # fig.savefig('/home/deepkliv/Saket/loss_plot.png')
    correct = 0
    total = 0
    print('\n Testing ....')
    autoencoder.train(False)  # For training
    classification.train(False)
    for t_i, t_data in enumerate(testloader):

      if t_i * batch_size > 1000:
        break
      t_inputs, t_labels = t_data
      t_inputs = Variable(t_inputs).cuda()
      t_labels = t_labels.cuda()
      t_outputs = autoencoder(t_inputs)
      c_pred = classification(t_inputs)
      _, predicted = torch.max(c_pred.data, 1)
      # print predicted.type() , t_labels.type()
      total += t_labels.size(0)
      correct += (predicted == t_labels).sum()
      if (epoch + 1) % 1 == 0:
        print("saving image")
        test_result_path = os.path.join('/home/deepkliv/Desktop/AE/ram/AE_classifier/fashion/Test_results/',
                                        'batch_%d_%d' % ((epoch + 1) / 1, t_i + 1) + '.jpg')
        image_tensor = torch.cat((t_inputs.data[0:8], t_outputs.data[0:8]), 0)
        torchvision.utils.save_image(image_tensor, test_result_path)

    print('Accuracy of the network on the 8000 test images: %d %%' % (100 * correct / total))

print('Finished Training and Testing')

"""
classification_criterion = nn.NLLLoss()
for i,data in enumerate(testloader):
	inputs,labels = data
	inputs,labels = Variable(inputs).cuda(), Variable(labels).cuda()
	test_model = Autoencoder().cuda()
	test_model.load_state_dict(torch.load('/home/deepkliv/Saket/AE_Classifier/Autoencoder/autoencoder-500.pkl',map_location=lambda storage, loc: storage.cuda(1)))
	outputs = test_model(inputs)
	test_result_path = os.path.join('/home/deepkliv/Saket/AE_Classifier/Test_results/', 'batch_%d'%(i+1) + '.jpg')
	image_tensor = torch.cat((inputs.data[0:8], outputs.data[0:8]), 0)
	torchvision.utils.save_image(image_tensor, test_result_path)
"""