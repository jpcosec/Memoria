'''Train CIFAR10 with PyTorch.'''

from lib.utils.debug import fake_arg


def kd_noise_sh(exp_name=""):
  f = open(exp_name + ".sh", "a")

  for student in ["ResNet18", "MobileNet"]:
    for distillation in ["KD"]:
      for T in [str(i) for i in [8]]:#[1, 5, 8, 10, 50, 100]]:
        for sigma in [0.1 * i for i in range(1, 11)]:
              
          transform = "noise," + str(sigma)
          dist = distillation + ",T-" + T
          st = f'python kd_distillation.py '\
              f'--student={student} ' \
              f'--distillation={dist} ' \
              f'--transform={transform} ' \
              f'--exp_name={transform.replace(",", "/")} \n'
    
          f.write(st)

  f.close()

def kd_sh(exp_name="",dataset="cifar10"):
  f = open(exp_name + ".sh", "a")

  for student in ["ResNet18", "MobileNet"]:
    for distillation in ["KD", "KD_CE"]:
      for T in [str(i) for i in [8]]:#[1, 5, 8, 10, 50, 100]]:
        #os.chdir("/home/jp/Memoria/repo/Cifar10/ResNet101/exp8"
         #        "")  # funcionalizar
        dist = distillation + ",T-" + T
        st = f'python kd_distillation.py '\
            f'--student={student} ' \
            f'--distillation={dist} ' \
            f'--dataset={dataset} ' \
            f'--exp_name={exp_name} \n'

        f.write(st)

  f.close()


if __name__ == '__main__':
    #args = fake_arg()

    #kd_sh("KD_normal", dataset=None)
    #kd_sh("KD_GAN",dataset="GAN")
    kd_noise_sh("kd_noise")