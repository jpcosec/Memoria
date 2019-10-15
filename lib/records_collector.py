import glob
import json

list_jsons=glob.glob("Cifar10/**/*.json", recursive=True)

for file_name in list_jsons:

  try:
    with open(file_name, 'r') as fp:
      record = json.load(fp)
      print(record)
      epoch = record["epoch"]
      train_step = record["train_step"]
      test_step = record["test_step"]

      print(file_name, epoch)

  except:
    print(file_name, "abrir wea" )

