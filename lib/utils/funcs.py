import os


def auto_change_dir(path):
  print("Moving to", path)#todo: log
  for folder in path.split("/"):
    if not os.path.exists(folder):
      print("Creating", folder)
      os.mkdir(folder)
    os.chdir(folder)


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