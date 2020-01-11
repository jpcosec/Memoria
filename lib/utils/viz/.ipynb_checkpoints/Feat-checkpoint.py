import pandas as pd
import altair as alt

def load_data(folder):
    source = pd.read_csv(folder+"/summary.csv") 
    source = source.fillna(1)
    source['student last_layer']=[row['student']+","+row['last_layer'] for i,row in source.iterrows()]
    return source

def plot(data,phase,field, center=False, scale=None):
    detalle=['test_acc', 'test_teacher/acc', 'test_loss', 'test_eval',
           'train_acc', 'train_teacher/acc', 'train_loss', 'train_eval','distillation','last_layer','layer']
    
    
    if scale is None:
        scale='log' if field in ['loss','eval'] else 'linear'
    else:
        print("scale", scale)
    
    field_dict={'acc':"Accuracy [%]", 'eval':"Perdida Cross Entropy", 'loss':"Perdida de Destilación"
        
    }
    
    data=data[data['distillation']!='nst_gauss']
    #data['train_acc']-=data['test_acc']
    
    
    bar=alt.Chart(data).mark_point().encode(
        alt.X('layer:O', scale=alt.Scale(zero=False,base=10),title="Capa"),
        alt.Y('%s_%s'%(phase,field), 
              scale=alt.Scale(zero=False, type=scale), 
              title=field_dict[field]),
        color=alt.Color('student last_layer'if center else 'student', legend=alt.Legend(title="Modelo, KD" if center else "Modelo")),
        fill=alt.Color('student last_layer'if center else 'last_layer', legend=alt.Legend(title="Modelo, KD" if center else "Destilacion Logits")),
        shape=alt.Color('distillation', legend=alt.Legend(title="Destilacion feat.")),
        size=alt.value(50),
        tooltip=detalle
    ).interactive()
    
    if field=='acc':
        accs = {'Model':['MobileNet','ResNet18','ResNet101'],
                      'ce_train':[95.73,98.15,98.52],
                      'ce_test':[87.8,90.58,90.68]}
                                 
        
        
        df=pd.DataFrame(accs)
        
        if center:
            d=dict(list(zip(accs['Model'],accs['ce_%s'%phase])))
            data['%s_acc'%phase]-=[d[i] for i in data['student']]
        
        else:
            aggregates = alt.Chart(df).mark_rule(opacity=0.5).encode(
                        y='ce_%s:Q'%phase,
                        color='Model:N',
                        size=alt.value(2))

        
            return (aggregates+bar).properties(width=600,height=400) 
    return bar.properties(width=600,height=400)


def load_and_plot(folder="./Cifar10/ResNet101/exp7/students",phase="test",field="acc",**kwargs):
    return plot(load_data(folder),phase,field,kwargs)


def omniplot(folder="./Cifar10/ResNet101/exp7/students",scale='log'):
    data = load_data(folder)

    detalle=['test_acc', 'test_loss', 'test_eval',
               'train_acc','train_loss', 'train_eval']
    chart=alt.Chart(data).mark_point().encode(
        alt.X(alt.repeat("column"), 
              type='quantitative',
              scale=alt.Scale(zero=False,base=10,type=scale)),
        alt.Y(alt.repeat("row"), 
              type='quantitative',
              scale=alt.Scale(zero=False,base=10,type=scale)),
        color='student last_layer',
        shape='distillation'
    ).properties(
        width=150,
        height=150
    ).repeat(
        row=detalle,
        column=detalle
    )
    
    return chart