import glob
import json
import pandas as pd

def maj_key(d):
    maj = -99999999999999
    for i in d.keys():
        try:
            num = int(i)
            maj = num if num > maj else maj
        except:
            pass
    if maj == -99999999999999:
        raise TypeError("No numerical key in dict")
    else:
        return maj



if __name__ == '__main__':
    #folder="/home/jp/Memoria/repo/Cifar10/ResNet101/exp1/students"
    folder = "/home/jp/Memoria/repo/Cifar10/ResNet101/students"
    list_jsons = glob.glob(folder+"/**/record.json", recursive=True)
    #list_jsons = glob.glob("/home/jp/Memoria/repo/Cifar10/ResNet101/student/**/record.json", recursive=True)
    info = {}

    def add_entry(name, field):
        if name not in info.keys():
            info[name] = []
        info[name].append(field)


    for file_name in list_jsons:


        with open(file_name, 'r') as fp:
            record = json.load(fp)
        #with open(file_name.replace("record", 'config'), 'r') as fp:
        #    config = json.load(fp)

        k = maj_key(record["test"])
        for key, value in record["test"][str(k)].items():
            add_entry("test_" + key, value)
        for key, value in record["train"][str(k)].items():
            add_entry("train_" + key, value)

        add_entry('epoch', record['epoch'])

        dist=file_name.split("/")[-2].split("-")
        add_entry('lambda', float(dist[1]))
        add_entry('layer', int(dist[2]))

        #add_entry('student',config['student'])
        #add_entry('distillation',config['distillation'].split(",")[0])

        #temp = float(config['distillation'].split(",")[1].replace("T-",""))
        #print(temp)
        #add_entry('temp', temp)

    csv=pd.DataFrame.from_dict(info)
    csv.to_csv(folder+"/summary.csv")
