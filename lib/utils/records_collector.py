import glob
import json


def maj_key(d):
    maj=-99999999999999
    for i in d.keys():
        try:
            num = int(i)
            maj = num if num > maj else maj
        except:
            pass
    if maj==-99999999999999:
        raise TypeError("No numerical key in dict" )
    else:
        return maj

if __name__ == '__main__':
    list_jsons = glob.glob("/home/jp/Memoria/repo/Cifar10/ResNet101/student/**/record.json", recursive=True)
    for file_name in list_jsons:

        try:
            with open(file_name, 'r') as fp:
                record = json.load(fp)
                k=maj_key(record["test"])
                last_record=record["test"][str(k)]
                print(last_record["acc"]-last_record["teacher/acc"])
        except:
            print(file_name, "abrir wea1")




        try:
            with open(file_name.replace("record",'config'), 'r') as fp:
                record = json.load(fp)
                print(record['student'])
                print(record['distillation'])
        #       print(record["test"])#["acc"] - record["test"]["teacher/acc"])
        except:
            print(file_name, "abrir wea")