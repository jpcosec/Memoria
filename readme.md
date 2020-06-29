# Feature Distillation 
Repo with several Feature Distillation implementations.
The repo contains a base library for experimentation with any model/dataset combination.
It was succesfully proven with a Cifar10 and Imagenet, and Cifar10 based synthetic datsets.
The results of the research are in Results.ipynb


# Execution

For a succesfull ussage of the dataparallel wrapper, it must be used in console and with linux.
I


* For experiments with Cifar10 use KD_distillation.py or feat_distillation.py. 
* For experiments with imageNet use imagenet_KD.py or imagenet_feats.py (KD means distillation using only logits and feats using logits and features). 
* For general experiments use example.py.

### Models
For Cifar10 experiments use teacher_train for training the teacher. 
There are several avaliable models in lib/models. 
Also, in lib/Artificial_Dataset_generators there are some models for synthetic datasets generation.


### Datasets

All important code is in folder lib. Cifar-10 models are in lib/models, imagenet models are downloaded directly using the torchvision API.  



# TODO: Escribir esto


**Document in https://www.overleaf.com/read/jsryrgvgyynd**
