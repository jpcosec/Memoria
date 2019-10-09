
import torch

import torch.backends.cudnn as cudnn

import os
from lib.utils import experiment
from lib.teacher.utils import load_model, get_model



class distillation_experiment():# TODO: solucionar problemas de herencia
  def __init__(self,**kwargs):

    self.student=kwargs["student"]
    self.teacher=kwargs["teacher"]
    self.eval_criterion=kwargs["eval_criterion"]
    self.device = kwargs["device"]
    self.optimizer = kwargs["optimizer"]
    self.criterion = kwargs["criterion"]
    self.flatten = kwargs["linear"]
    self.writer = kwargs["writer"]
    self.testloader = kwargs["testloader"]
    self.trainloader = kwargs["trainloader"]
    self.best_acc = kwargs["best_acc"]
    self.net = self.student
# class distiller()

def load_teacher(args,device):

  print('==> Building teacher model..', args.teacher)
  net = get_model(args.teacher)
  net = net.to(device)

  for param in net.parameters():
    param.requires_grad = False

  if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


  assert os.path.isdir(args.teacher), 'Error: model not initialized'
  os.chdir(args.teacher)
  # Load checkpoint.
  print('==> Resuming from checkpoint..')
  assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
  checkpoint = torch.load('./checkpoint/ckpt.pth')
  net.load_state_dict(checkpoint['net'])

  return net

def load_student(args,device):
  best_acc = 0  # best test accuracy
  start_epoch = 0  # start from epoch 0 or last checkpoint epoch
  folder="student/"+args.student+"/"+args.distillation
  # Model
  print('==> Building student model..', args.student)
  net = get_model(args.student)
  net = net.to(device)
  if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

  if args.resume:
    assert os.path.isdir(folder), 'Error: model not initialized'
    os.chdir(folder)
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['student_acc']
    start_epoch = checkpoint['epoch']

    if start_epoch >= args.epochs:
      print("Number of epochs already trained")
  else:
    if not os.path.isdir("student/"):
      os.mkdir("student")
    if not os.path.isdir("student/"+args.student):
      os.mkdir("student/"+args.student)
    os.mkdir(folder)
    os.chdir(folder)
  return net, best_acc, start_epoch


def train(exp, epoch):
  print('\rTraining epoch: %d' % epoch)
  exp.student.train()
  exp.teacher.eval()
  total_loss = 0
  correct = 0
  total = 0
  for batch_idx, (inputs, targets) in enumerate(exp.trainloader):  # 784= 28*28
    inputs, targets = inputs.to(exp.device), targets.to(exp.device)
    exp.optimizer.zero_grad()

    # Predecir
    if exp.flatten:
      S_y_pred = exp.student(inputs.view(-1, 3072))
    else:
      S_y_pred = exp.student(inputs)



    T_y_pred = exp.teacher(inputs)

    # Compute Loss
    loss = exp.criterion(S_y_pred, T_y_pred)
    # Backward pass
    loss.backward()
    exp.optimizer.step()

    total_loss += loss.item()
    _, predicted = S_y_pred.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    loss = total_loss / (batch_idx + 1)

    EC = exp.eval_criterion(S_y_pred, targets).item()

    exp.writer.add_scalar('train/loss', loss)
    exp.writer.add_scalar('train/acc', acc)
    exp.writer.add_scalar("train/EvalCriterion",EC)

  print("loss=",loss)
  print("EC=", EC)
  print("Acc=", acc)



def test(exp, epoch):
  print('\rTesting epoch: %d' % epoch)
  exp.student.eval()
  exp.teacher.eval()




  ac_loss = 0
  student_correct = 0
  teacher_correct = 0
  total = 0

  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(exp.testloader):
      inputs, targets = inputs.to(exp.device), targets.to(exp.device)
      # Predecir
      if exp.flatten:
        S_y_pred = exp.student(inputs.view(-1, 3072))
      else:
        S_y_pred = exp.student(inputs)


      T_y_pred = exp.teacher(inputs)
      dist_loss = exp.criterion(S_y_pred, T_y_pred)

      student_eval = exp.eval_criterion(S_y_pred.squeeze(), targets)
      teacher_eval = exp.eval_criterion(T_y_pred.squeeze(), targets)

      ac_loss += dist_loss.item()


      _, predicted = S_y_pred.max(1)
      total += targets.size(0)
      student_correct += predicted.eq(targets).sum().item()

      _, predicted = T_y_pred.max(1)
      teacher_correct += predicted.eq(targets).sum().item()



      student_acc = 100. * student_correct / total
      teacher_acc = 100. * teacher_correct / total
      loss=ac_loss/total

      #progress_bar(batch_idx, len(exp.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
      #             % (ac_loss / (batch_idx + 1), 100. * student_correct / total, student_correct, total))



      exp.writer.add_scalar('test/student/acc', student_acc)
      exp.writer.add_scalar('test/teacher/acc', teacher_acc)
      exp.writer.add_scalar('test/ac_loss', loss)

      exp.writer.add_scalar("test/student/eval",student_eval)
      exp.writer.add_scalar("test/teacher/eval", teacher_eval)

  # Early stoping, # Save checkpoint.
  if student_acc > exp.best_acc:
    print('Saving..')
    state = {
      'net': exp.net.state_dict(),
      'student_acc': student_acc,
      'epoch': epoch
    }
    if not os.path.isdir('checkpoint'):
      os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.pth')
    exp.best_acc = student_acc

  print("ac_loss=",ac_loss)
  print("Student_EC=", student_eval)
  print("Teacher_EC=", teacher_eval)
  print("Student_Acc=", student_acc)
  print("Teacher_Acc=", student_acc)
