'''Train CIFAR10 with PyTorch.'''

from lib.utils.debug import fake_arg

blocs = {"ResNet101": [26, 56, 219, 239],  # Completar
         "MobileNet": [6, 15, 26, 55],
         "ResNet18": [10, 23, 35, 46]
         }


def make_noise_sh(exp_name=""):
  f = open(exp_name + ".sh", "a")
  for student in ["ResNet18",
                  "MobileNet"]:  # todo: terminar nst poly 3 y hint 1 desde 0"MobileNet", Hint3 en resnet (y 1 si no hayrecupere)
    for distillation in ["nst_linear",
                         # "nst_poly",
                         "att_mean",
                         "att_max",
                         "hint",
                         "PKT"]:
      for layer, (s_layer, t_layer) in enumerate(zip(blocs[student], blocs["ResNet101"])):
        for sigma in [0.1 * i for i in range(1, 11)]:
          # os.chdir("/home/jp/Memoria/repo/Cifar10/ResNet101/"+folder)

          arg = fake_arg(distillation=distillation,
                         student=student,
                         layer=layer,
                         student_layer=s_layer,
                         teacher_layer=t_layer,
                         )


          transform = "noise," + str(sigma)

          st = f'python feat_distillation.py ' \
            f'--distillation={distillation} ' \
            f'--layer={layer} ' \
            f'--student={student} ' \
            f'--student_layer={s_layer} ' \
            f'--teacher_layer={t_layer} ' \
            f'--transform={transform} ' \
            f'--exp_name={transform.replace(",", "/")} \n'
          f.write(st)

  f.close()



def make_sh(exp_name="",dataset="cifar10"):
  f = open(exp_name + ".sh", "a")
  dist_list=["nst_linear",
             # "nst_poly",
             "att_mean",
             "att_max",
             "hint",
             "PKT"]

  for student in ["ResNet18",
                  "MobileNet"]:
    for distillation in dist_list:
      for layer, (s_layer, t_layer) in enumerate(zip(blocs[student], blocs["ResNet101"])):

          st = f'python feat_distillation.py ' \
            f'--distillation={distillation} ' \
            f'--layer={layer} ' \
            f'--student={student} ' \
            f'--student_layer={s_layer} ' \
            f'--teacher_layer={t_layer} ' \
            f'--dataset={dataset} ' \
            f'--exp_name={exp_name} \n'

          f.write(st)

  f.close()


def kd_sh(exp_name="",dataset="cifar10"):
  f = open(exp_name + ".sh", "a")

  for student in ["ResNet18", "MobileNet"]:
    for distillation in ["KD", "KD_CE"]:
      for T in [str(i) for i in [1, 5, 10, 50, 100, 1000]]:
        #os.chdir("/home/jp/Memoria/repo/Cifar10/ResNet101/exp8"
         #        "")  # funcionalizar
        dist = distillation + ",T-" + T
        st = f'python kd_distillation.py '\
            f'--distillation={dist} ' \
            f'--dataset={dataset} ' \
            f'--exp_name={exp_name} \n'

        f.write(st)

  f.close()


if __name__ == '__main__':
    #args = fake_arg()



    kd_sh("KD_GAN")#,dataset="")