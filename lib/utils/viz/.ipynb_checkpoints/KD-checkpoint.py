import pandas as pd
import altair as alt
def load_data(folder="./Cifar10/ResNet101/exp1/students"):
    data = pd.read_csv(folder+"/summary.csv") 
    data = data[data["student"] != 'EfficientNetB0']
    if "temp" not in data.columns:
        d_arr=[d.split(",")[0] for d in data["distillation"]]
        t_arr=[float(d.split(",T-")[1]) for d in data["distillation"]]
        data["distillation"]=d_arr
        data["temp"]=t_arr
    return data


def plot(data,phase,field,center=False,scale=None,shape=(400,400)):
    detalle=['test_acc', 'test_teacher/acc', 'test_loss', 'test_eval',
           'train_acc', 'train_teacher/acc', 'train_loss', 'train_eval','distillation', 'temp']
    
    field_dict={'acc':"Accuracy [%]", 'eval':"Perdida Cross Entropy", 'loss':"Perdida de Destilación"
        
    }
    
    if scale is None:
        print(".")
        scale='log' if field in ['loss','eval'] else 'linear'
    else:
        print("scale", scale)
    
    print("lasorra")
    #data['train_acc']-=data['test_acc']
    
    
    bar=alt.Chart(data).mark_point().encode(
        alt.X('temp:O', scale=alt.Scale(zero=False,base=10,type='log', ),title="Temperatura"),
        alt.Y('%s_%s'%(phase,field), 
              scale=alt.Scale(zero=False, type=scale), 
              title=field_dict[field]),
        shape=alt.Shape('distillation', legend=alt.Legend(title="Destilación")),
        color=alt.Color('student', legend=alt.Legend(title="Modelo")),
        size=alt.value(50),
        tooltip=detalle
    ).interactive()
    
    if field == 'acc':
        accs = {'Model':['MobileNet','ResNet18','ResNet101'],
                      'ce_train':[95.73,98.15,98.52],
                      'ce_test':[87.8,90.58,90.68]}
        if center:
            d=dict(list(zip(accs['Model'],accs['ce_%s'%phase])))
            data['%s_acc'%phase]-=[d[i] for i in data['student']]        
        else:
            df=pd.DataFrame(accs)
            aggregates = alt.Chart(df).mark_rule(opacity=0.5).encode(
                        y='ce_%s:Q'%phase,
                        color='Model:N',
                        size=alt.value(2))
            
            return (aggregates+bar).properties(width=shape[0],height=shape[1]) 
        
    return bar.properties(width=shape[0],height=shape[1])
    
def load_and_plot(folder="./Cifar10/ResNet101/exp1/students",phase='test',field='acc',**kwargs):
    data = load_data(folder)

    return plot(data,phase,field,**kwargs)

def omniplot(folder="./Cifar10/ResNet101/exp1/students"):
    data = load_data(folder)

    detalle=['test_acc', 'test_loss', 'test_eval',
               'train_acc','train_loss', 'train_eval', 'temp']
    chart=alt.Chart(data).mark_point().encode(
        alt.X(alt.repeat("column"), type='quantitative',scale=alt.Scale(zero=False,base=10,type='log')),
        alt.Y(alt.repeat("row"), type='quantitative',scale=alt.Scale(zero=False,base=10,type='log')),
        shape='student',
        color='distillation'
    ).properties(
        width=150,
        height=150
    ).repeat(
        row=detalle,
        column=detalle
    )
    return chart