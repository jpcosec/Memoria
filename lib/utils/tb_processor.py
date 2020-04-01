from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import glob
from lib.utils.funcs import auto_make_dir
def extract_tb_scalars(exp,folder,file):

  ea = event_accumulator.EventAccumulator(exp+folder, file)

  ea.Reload()

  folder = "arrs_collect/" + folder.replace("tb_logs/", "").replace("students/", "")
  auto_make_dir(folder)

  try:

    train_kw=['train/acc',
              'train/loss',
              'train/eval']
    train_dict=dict([(kw,np.array([i.value for i in ea.Scalars(kw)])) for kw in train_kw] )

    train_dict["train/wall_time"]=np.array([i.wall_time for i in ea.Scalars("train/acc")])


    test_kw=['test/acc',
             'test/loss',
             'test/eval']
    test_dict=dict([(kw,np.array([i.value for i in ea.Scalars(kw)])) for kw in test_kw] )
    test_dict["test/wall_time"]=np.array([i.wall_time for i in ea.Scalars("test/acc")])

    np.savez_compressed("./"+folder+"arrs_comp",**test_dict,**train_dict)
  except:
    print(folder)



def scallars_collector(folder):
  tb_logs = glob.glob(folder + "/**/tb_logs/", recursive=True)
  for path in tb_logs:
    filez=glob.glob(path+"/*")
    if len(filez)>1:
      print("problwmo")
    else:
      file=filez[0].replace(path,"")
      extract_tb_scalars(folder,path.replace(folder,""),file)

if __name__ == '__main__':
    scallars_collector("./Cifar10/")
    #auto_make_dir("la/so/rra")

    #list_jsons = glob.glob("../Cifar10/" + "/**/record.json", recursive=True)