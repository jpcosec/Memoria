import pandas as pd
import numpy as np

import json

import glob

global_titles={"test_acc":"Accuracy en validacion",
               'test/acc':"Accuracy en validacion",
               'test/wall_time':"Tiempo de entrenamiento [s]",
                "test_teacher/acc":"Accuracy de red tutora en validacion",
                "test_loss":"Perdida en entrenamiento",
                "test_eval":"Evalucacion en entrenamiento",
                "train_acc":"Accuracy en entrenamiento",
                "train_teacher/acc":"Accuracy de red tutora en entrenamiento",
                "train_loss":"Perdida en entrenamiento",
                "train_eval":"Evaluacion en entrenamiento",
                "student":"Modelo",
                "model":"Modelo",
                "layer":"Bloque",
                "transform":"Transformacion",
                "dataset":"Base de datos",
                "temp":"Temperatura",
                "log_dist":"Perdida en logits",
                "feat_dist":"Perdida en features",
                "delta_teacher":"Diferencia con red tutora en entrenamiento",
                "delta_student":"Diferencia con red estudiante (CE) en entrenamiento",
                "test/train":"Ratio de accuracy en entreamiento y validacion",
                "noise":"Potencia de Ruido",
                "loss_conv":"Epoca de de minima perdida alcanzada",
                "acc_conv":"Epoca de maximo accuracy alcanzado",
                "max":"Accuracy en validacion",
                "epochs":'Epoca',
                'dists': 'Destilacion en features, en logits',
                'feat,block': 'Destilacion en features, bloque',
                'sint_key': 'Tipo de dataset'
                }

"""
    Data Processing
"""

CE_path = ["Cifar10/%s/record.json" % model for model in ["ResNet101", "ResNet18", "MobileNet"]]
paths = glob.glob("Cifar10/**/summary.csv", recursive=True)

def load_teachers(CE_path):
    def flatten_record(path):
        i = json.load(open(path, 'r'))
        d = {}
        for k, v in i.items():
            if isinstance(v, dict):
                for l in v.values():
                    for ki, vi in l.items():
                        # print(l.values())
                        d["_".join([k, ki])] = vi
            else:
                d[k] = v
        return d

    d = {'model': [],
         'test_acc': [],
         'test_loss ': [],
         'train_acc': [],
         'train_loss ': [],
         'epoch': [],
         'train_step': [],
         'test_step': []}

    for path in CE_path:
        d['model'].append(path.split("/")[-2])
        for k, v in flatten_record(path).items():
            d[k].append(v)
    return pd.DataFrame.from_dict(d)


data = {}
data['teachers'] = load_teachers(CE_path)
data['teachers'].index = (data['teachers']['model'])
data["students"] = {'feat': {},
                    'kd': {}}

drop_cols = ['Unnamed: 0', 'epoch', 'lr', 'resume', 'epochs', 'train_batch_size',
             'test_batch_size', 'batch_size', 'teacher', 'exp_name', 'student_layer',
             'teacher_layer', 'distillation', 'pre', 'test_eval', 'train_eval', 'test_teacher/acc',
             'train_teacher/acc']

# print(paths)
for path in paths:
    ks = path.split("/")[-2].split("_")
    da = pd.read_csv(path)

    cols = [i for i in da if i in drop_cols]

    if ks[0] == 'kd':
        da['temp'] = [int(i.split('-')[-1]) for i in da['distillation']]
        da['log_dist'] = [i.split(',')[0] for i in da['distillation']]
        da['feat_dist'] = '-'
        da['layer'] = '-'
    elif ks[0] == 'feat':

        da['temp'] = [int(float(i.split('-')[-1])) if len(i.split('-')) > 1 else 8 for i in da['last_layer']]
        da['log_dist'] = [i.split(',')[0] for i in da['last_layer']]
        da['feat_dist'] = da['distillation']

    da['transform'] = ['-' if i == 'none,' else i for i in da['transform']]
    da['delta_teacher'] = da['test_acc'] - data["teachers"].at['ResNet101', 'test_acc']
    da['test/train'] = da['test_acc'] / da["train_acc"]
    da['noise'] = [float(i.split(",")[1]) if i != '-' else 0.0 for i in da['transform']]

    da['dataset'] = [i.replace(",", "").replace("-Dataset", "") for i in da['dataset']]
    if ks[0] == 'kd' and ks[1] == 'normal':
        da['dataset'] = 'cifar10'

    if ks[0] == 'feat' and ks[1] == 'KD-CE':
        da['dataset'] = 'cifar10'
        ks[1] = 'KD_CE'

    da['delta_student'] = [r['test_acc'] - data['teachers'].at[r['student'], 'test_acc'] for i, r in da.iterrows()]
    da.index = pd.Index([",".join([str(i) for i in
                                   [row["student"], row["dataset"], row["log_dist"], row["temp"], row["feat_dist"],
                                    row["layer"], row["noise"]]]) for i, row in da.iterrows()])

    data["students"][ks[0]]['KD' if ks[1] == 'normal' else ks[1]] = da.drop(cols, axis=1)

data['students']['kd']['KD']['dataset'] = 'cifar10'

import json

"""
    Datos en tiempo
"""

epoch_data = json.load(open('Cifar10/epoch_summary.json', 'r'))


def exp_parse(split):
    temp = 8
    split = k.split("/")
    exp = split[2]
    exp_type = "feat"
    if len(exp.split("_")) > 1:
        if exp.split("_")[0].lower() == 'kd':
            exp_type = "kd"

    if exp_type == "kd":
        dataset = exp.split("_")[1]
    elif len(exp.split("_")) == 1:
        dataset = exp
    elif exp.split("_")[0] == "feat":
        dataset = exp.split("_")[1]
    else:
        dataset = exp.split("_")[0]
    if dataset == 'normal':
        dataset = 'KD'

    if dataset == "noise":
        # transform=
        # ['Cifar10', 'ResNet101', 'noise', '0.1', 'MobileNet', 'att_max', '0', '', '']
        noise = float(split[3])
        model = split[4]
        if exp_type == 'kd':
            temp = int(split[6].split("-")[-1])
            logit_loss = split[5]
            # print(model,logit_loss,dataset, temp,noise)
        else:
            layer = int(split[6])
            feat_loss = split[5]
            # print(model,feat_loss,dataset, layer,noise)
    else:
        noise = 0.0
        model = split[3]
        if exp_type == 'kd':
            temp = int(split[5].split("-")[-1])
            logit_loss = split[4]
        else:
            layer = int(split[5])
            feat_loss = split[4]

    if exp_type == 'kd':
        feat_loss = '-'
        layer = '-'
    else:
        logit_loss = dataset if dataset not in ['GAN', 'VAE', 'noise'] else 'KD'
    key2 = dataset
    dataset = dataset if dataset in ['GAN', 'VAE'] else "cifar10"

    return model, exp_type, logit_loss, feat_loss, key2, layer, noise, dataset, temp


conv_data = {}
conv_data['teachers'] = {'loss_conv': [], 'acc_conv': [], 'index': [], 'max': [], 't': []}  # load_teachers(CE_path)
conv_data["students"] = {'feat': dict([(k, {'loss_conv': [], 'acc_conv': [], 'index': [], 'max': [], 't': []}) for k in
                                       list(data['students']['feat'].keys())]),
                         'kd': dict([(k, {'loss_conv': [], 'acc_conv': [], 'index': [], 'max': [], 't': []}) for k in
                                     list(data['students']['kd'].keys())])}


def convergencia(s, neg=True):
    j = [s[i] - s[i - 1] for i in range(1, len(s))]
    p = 3
    for n, i in enumerate(j):
        if (i > 0) == neg:
            p -= 1
        else:
            p = 3

        if p < 0:
            return n - 2


for k, v in epoch_data.items():
    split = k.split("/")
    teacher = split[1]
    exp = split[2]
    # print(exp)
    if exp == 'teacher':
        # teacher_dict[teacher]=v
        conv_data['teachers']['index'].append(teacher)
        conv_data['teachers']['loss_conv'].append(np.argmin(v["train/loss_"]))
        conv_data['teachers']['acc_conv'].append(np.argmax(v["test/acc"]))
        conv_data['teachers']['max'].append(np.max(v["test/acc"]))
        conv_data['teachers']['t'].append(k)
    else:
        model, exp_type, logit_loss, feat_loss, key2, layer, noise, dataset, temp = exp_parse(k)  # (FALTA UNA KEY
        logit_loss = logit_loss.replace("-", "_")
        key = ','.join([str(i) for i in [model, dataset, logit_loss, temp, feat_loss, layer, noise]])
        key2 = key2.replace("-", "_")

        conv_data['students'][exp_type][key2]['index'].append(key)
        conv_data['students'][exp_type][key2]['loss_conv'].append(np.argmin(v["train/loss"]))
        conv_data['students'][exp_type][key2]['acc_conv'].append(np.argmax(v["test/acc"]))
        conv_data['students'][exp_type][key2]['max'].append(np.max(v["test/acc"]))
        conv_data['students'][exp_type][key2]['t'].append(k)

# FALTA TEACHER
for k in list(data['students']['feat'].keys()):
    i = conv_data['students']['feat'][k]
    p = pd.DataFrame(data=(i), index=i['index']).drop(columns=['index'])
    data['students']['feat'][k] = data['students']['feat'][k].join(p.reindex(data['students']['feat'][k].index))

for k in list(data['students']['kd'].keys()):
    i = conv_data['students']['kd'][k]
    p = pd.DataFrame(data=(i), index=i['index']).drop(columns=['index'])
    data['students']['kd'][k] = data['students']['kd'][k].join(p.reindex(data['students']['kd'][k].index))

reidx = ['dataset', 'transform', 'noise', 'student',
         'log_dist', 'temp', 'feat_dist', 'layer', 'delta_teacher',
         'delta_student', 'test/train', 'max', 'test_acc', 'test_loss',
         'train_acc', 'train_loss', 't', 'loss_conv', 'acc_conv', ]

for k in list(data['students']['feat'].keys()):
    data['students']['feat'][k] = data['students']['feat'][k].reindex(reidx, axis='columns').reset_index(drop=True)
for k in list(data['students']['kd'].keys()):
    data['students']['kd'][k] = data['students']['kd'][k].reindex(reidx, axis='columns').reset_index(drop=True)

i = conv_data['teachers']
p = pd.DataFrame(data=(i), index=i['index']).drop(columns=['index'])
data['teachers'] = data['teachers'].join(p.reindex(data['teachers'].index))

data["teachers"]['delta_teacher'] = [i - data["teachers"].at['ResNet101', 'test_acc'] for i in
                                     data["teachers"]['test_acc']]
data["teachers"]['delta_student'] = 0
data["teachers"]['test/train'] = data["teachers"]['test_acc'] / data["teachers"]["train_acc"]

data["teachers"]['student'] = data["teachers"]['model']
data["teachers"]['log_dist'] = 'CE'
data["teachers"]['feat_dist'] = '-'
data["teachers"]['layer'] = '-'

data["teachers"]['train_loss'] = data["teachers"]['train_loss ']
data["teachers"]['test_loss'] = data["teachers"]['test_loss ']
data['teachers'] = data['teachers'].drop(columns=['test_loss ', 'train_loss '])
bar_source = data["teachers"][data["teachers"]['model'] != "ResNet101"].copy()



"""
    Utils
"""


def normal_whole_source(key):
    if key == 'CE':
        source=data['students']['feat']['CE']
        drop_cols="model epoch 	train_step 	test_step".split()
        
        source=pd.concat([bar_source.drop(columns=drop_cols),
                          source])
        source['temp']='-'
        return source
    elif key == 'KD_CE':
            d=data['students']['kd']['KD']
            d=d[d["temp"]==8]
            d=d[d["log_dist"]==key]
            source=data['students']['feat'][key]
            source=pd.concat([d,source])
            return source
    else:
        d=data['students']['kd'][key]
        d=d[d["temp"]==8]
        d=d[d["log_dist"]==key]
        source=data['students']['feat'][key]
        source=pd.concat([d,source])
        
    return source

def noise_load():
    s1=normal_whole_source('KD')
    #s1=s1[s1['feat_dist']!='nst_poly']
    s2=data['students']['feat']['noise']
    s3=data['students']['kd']['noise']
    source=pd.concat([s1[s1['feat_dist']!='nst_poly'],s2,s3])
    

    return source

# datasets de tiempo
def d2plambda( dic,model):
    #for k,i in dic.items():
    #    dic[k]=[0]
        
    df=pd.DataFrame.from_dict(dic)
    df['model']=model
    df['epochs']=range(1,len(df['model'])+1)
    #df
    return df
