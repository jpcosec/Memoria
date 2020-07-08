# Feature Distillation 
The repo contains a base Class for feature distillation experimentation with any model/dataset combination,
 right now the Distillator class allows the ussage of only one type of simultaneous distillation with 1 layer per time, 
 it can be easilly modified on lib/feature_distillators.


The repo was succesfully proven with Cifar10, Imagenet and Cifar10 based synthetic datsets.
The results of the research are in Results.ipynb




# Execution

For a succesful ussage of the dataparallel wrapper, it must be used in console and with linux.
For running several experiments at once, a routine of experiments can be generated with the exp_generators.


* For experiments with Cifar10 use KD_distillation.py or feat_distillation.py. 
* For experiments with imageNet use imagenet_KD.py or imagenet_feats.py (KD means distillation using only logits and feats using logits and features). 
* For general experiments use example.py.


### Models
For Cifar10 experiments use teacher_train for training the teacher. 
There are several avaliable models in lib/models. 
Also, in lib/Artificial_Dataset_generators a VAE and a GAN model were implemented, for synthetic datasets generation.
Imagenet models are downloaded directly using the torchvision API.  

With little modifications the example.py script any model can be used.


### Datasets

Currently there are implementations for Cifar10 and Imagenet and a torchvision datafolder dataset implementation in 
example.py. The implementation of any other dataset can be implemented easily using the pytorch dataset class. 
In lib/Artificial_Dataset_generators there are examples for GAN and VAE synthetic datasets generators ussage.

**The whole document (in spanish) is in https://www.overleaf.com/read/jsryrgvgyynd**
