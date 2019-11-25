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


def collect_records(folder):
    # Collects all records and returns pandas shaped dict
    list_jsons = glob.glob(folder + "/**/record.json", recursive=True)

    info = {}

    def add_entry(name, field,i,d):
        if name not in info.keys():
            info[name] = []
            if i>0:
                for i in range(i):
                    info[name].append("")
        info[name].append(field)
        d[name]=False

    for i,file_name in enumerate(list_jsons):

        fill_dict=dict([(k, True) for k in info.keys()])

        with open(file_name, 'r') as fp:
            record = json.load(fp)

        with open(file_name.replace("record", 'config'), 'r') as fp:
            config = json.load(fp)

        k = maj_key(record["test"])
        for key, value in record["test"][str(k)].items():
            add_entry("test_" + key, value,i,fill_dict)
        k = maj_key(record["train"])
        for key, value in record["train"][str(k)].items():
            add_entry("train_" + key, value,i,fill_dict)

        add_entry('epoch', record['epoch'],i,fill_dict)

        for key, value in config.items():
            add_entry(key,value,i,fill_dict)


        for key, value in fill_dict.items():
            if value:
                info[key].append("")



    return info

if __name__ == '__main__':
    #folder="/home/jp/Memoria/repo/Cifar10/ResNet101/exp1/students"
    folder = "/home/jp/Memoria/repo/Cifar10/ResNet101/students/ResNet18"

    info=collect_records(folder)



    for key, value in info.items():
        print(key, len(value))

    #print(list(zip(info["layer"],info["lambda"],info["distillation"])))

    print(info)
    csv=pd.DataFrame.from_dict(info)
    csv.to_csv(folder+"/summary.csv")
    print("done")
