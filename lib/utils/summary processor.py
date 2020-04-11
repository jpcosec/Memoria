
from functools import reduce
import glob

def doc2Json(string):
  d = {'layers': {},
       'block_changes': []}
  layer_sec = False
  params_sec = False

  last_dim = 0
  block = 0

  for line in string.split("\n"):
    if line[0] == "=":
      if layer_sec:
        params_sec = True
      layer_sec = True
    else:
      if params_sec:
        if line[0] != "-":
          fields = line.split(":")
          d[fields[0]] = fields[1].replace(" ", "")
      else:
        if layer_sec:
          arr = line.replace(",", "").replace("[", "").replace("]", "").split()
          lt = arr[0].split("-")

          if last_dim != arr[-2]:
            block += 1
            last_dim = arr[-2]
            d["block_changes"].append(int(lt[-1]))

          di = {"id": int(lt[1]),
                "n_params": int(arr[-1]),
                "dimensions": arr[1:-1],
                "block": block}

          field = lt[0]
          if not field in d['layers'].keys():
            d['layers'][field] = []
          d['layers'][field].append(di)

  d["layer_count"] = dict([(key, len(value)) for key, value in d['layers'].items()])
  d["layer_count"]["total"] = reduce(lambda x, y: x + y, d["layer_count"].values())

  d["param_count"] = dict(
    [(key, reduce(lambda x, y: x + y, [i["n_params"] for i in value])) for key, value in d['layers'].items()])

  return d

if __name__ == '__main__':
  paths=glob.glob("/home/jpruiz/PycharmProjects/Memoria/docs/model_summaries/**/*.txt",recursive=True)
  #import os

  for path in paths:
    with open(path,'r') as f:
      print(path)
      string=f.read()
      d = doc2Json(string)
      d["layers"] = ""
      print(d)
