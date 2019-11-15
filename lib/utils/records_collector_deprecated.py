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
        print(file_name)
        k = maj_key(record["test"])
        for key, value in record["test"][str(k)].items():
            add_entry("test_" + key, value)
        k = maj_key(record["train"])
        for key, value in record["train"][str(k)].items():
            add_entry("train_" + key, value)

        add_entry('epoch', record['epoch'])


        dist = file_name.split("/")[-2].split("-")
        if dist[0]=="feature":
            print("lasorra")
            add_entry('layer', int(dist[2]))
            add_entry('distillation', dist[0])
            print(dist)
            d={"lr": 0.1,
             "resume": False,
             "epochs": 100,
             "pre": 50,
             "train_batch_size": 128,
             "test_batch_size": 100,
             "student": "ResNet18",
             "teacher": "ResNet101",
             "distillation": dist[0].replace("feature","hint").replace("attention","att_mean"),
             "layer": int(dist[2])}

            d["lambda"]=float(dist[1])

            with open(file_name.replace("record", 'config'), 'w') as outfile:
                json.dump(d, outfile)



        #add_entry('student',config['student'])
        #add_entry('distillation',config['distillation'].split(",")[0])

        #temp = float(config['distillation'].split(",")[1].replace("T-",""))
        #print(temp)
        #add_entry('temp', temp)

    csv=pd.DataFrame.from_dict(info)
    csv.to_csv(folder+"/summary.csv")
