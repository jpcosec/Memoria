from functools import reduce
import glob
from torchsummary import summary
import pandas as pd


def doc2Json(string):
    d = {'layers': {},
         'block_changes': []}
    layer_sec = False
    params_sec = False

    last_dim = 0
    block = 0

    blocks=[]

    di = {"id": None,
          "n_params": None,
          "dimensions": None,
          "block": None,
          "last_relu": None,
          "last_conv2d": None}

    for line in string.split("\n"):
        if line[0] == "=":
            if layer_sec:
                params_sec = True
            layer_sec = True
        else:
            if params_sec:
                """
                Bottom text description processing
                """
                if line[0] != "-":
                    fields = line.split(":")
                    d[fields[0]] = fields[1].replace(" ", "")
            else:
                if layer_sec:
                    # line parsing
                    arr = line.replace(",", "").replace("[", "").replace("]", "").split()
                    layer_info = arr[0].split("-")

                    if last_dim != arr[-2]:

                        d["block_changes"].append(int(layer_info[-1]))
                        blocks.append(di.copy())
                        #print(di)

                        block += 1
                        last_dim = arr[-2]

                        di["id"] = int(arr[-1].split(",")[-1])
                        di["n_params"] = int(arr[-1])
                        di["dimensions"] = arr[1:-1]
                        di["block"] = block


                    if layer_info[0][:4] == "ReLU":
                        di["last_relu"] = layer_info[1]

                    if layer_info[0] == 'Conv2d':
                        di["last_conv2d"] = layer_info[1]

                    field = layer_info[0]
                    if not field in d['layers'].keys():
                        d['layers'][field] = []
                    d['layers'][field].append(di)

    d["layer_count"] = dict([(key, len(value)) for key, value in d['layers'].items()])
    d["layer_count"]["total"] = reduce(lambda x, y: x + y, d["layer_count"].values())

    d["param_count"] = dict(
        [(key, reduce(lambda x, y: x + y, [i["n_params"] for i in value])) for key, value in d['layers'].items()])
    d['blocks'] = blocks[1:]

    d["layers"] = ""
    return d


if __name__ == '__main__':
    folder="/home/jpruiz/PycharmProjects/Memoria/docs/model_summaries/"
    paths = glob.glob(folder+"**/*.txt", recursive=True)
    # import os
    import pandas as pd
    import json

    block_json={}
    keys=["key",'layers', 'block_changes', 'Total params', 'Trainable params', 'Non-trainable params', 'Input size (MB)', 'Forward/backward pass size (MB)', 'Params size (MB)', 'Estimated Total Size (MB)', 'layer_count', 'param_count']
    data_pandas=dict([(k,[]) for k in keys])



    for path in paths:
        with open(path, 'r') as f:
            print(path)
            string = f.read()
            d = doc2Json(string)
            #print(d.keys())

            key=path.replace(folder,"")
            block_json[key]=d['blocks']
            d['blocks']=""
            d["key"]=key
            for k in keys:
                data_pandas[k].append(d[k])




    print(data_pandas)
    data=pd.DataFrame.from_dict(data_pandas)
    data.to_csv(folder+"summary.csv")
    print(block_json)
    json.dump(block_json,fp=open(folder+"wea.json","w"),indent=4)



