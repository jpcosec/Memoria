import os


def auto_change_dir(path):
  print("Moving to", path)#todo: log
  for folder in path.split("/"):
    if not os.path.exists(folder):
      print("Creating", folder)
      os.mkdir(folder)
    os.chdir(folder)

def auto_make_dir(path):
  print("creating", path)#todo: log

  temp_path=""
  for folder in path.split("/"):
    temp_path+=folder
    temp_path+="/"
    if not os.path.exists(temp_path):
      print("Creating", temp_path)
      os.mkdir(temp_path)


def register_hooks(net, idxs, feature):# Deprecate
  """
  Registers a hook in the module of the net
  :param net:
  :param idxs:
  :param feature:
  :return:
  """

  def hook(m, i, o):
    feature[m] = o

  for name, module in net._modules.items():
    for id, layer in enumerate(module.children()):
      if id in idxs:
        layer.register_forward_hook(hook)


def check_folders(folders=["outs","checkpoints"]):
  #print("Moving to", path)#todo: log
  for folder in folders:
    if not os.path.exists(folder):
      print("Creating", folder)
      os.mkdir(folder)