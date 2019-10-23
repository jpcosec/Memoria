import glob
import json

list_jsons = glob.glob("/home/jp/Memoria/repo/Cifar10/ResNet101/student/**/record.json", recursive=True)

for file_name in list_jsons:

    try:
        with open(file_name, 'r') as fp:
            record = json.load(fp)
            int_keys = [int(i) for i in record.keys()]
            print(int_keys)
    except:
        print(file_name, "abrir wea")




    try:
        with open(file_name.replace("record",'config'), 'r') as fp:
            record = json.load(fp)
            print(record['student'])
            print(record['distillation'])
    #       print(record["test"])#["acc"] - record["test"]["teacher/acc"])
    except:
        print(file_name, "abrir wea")