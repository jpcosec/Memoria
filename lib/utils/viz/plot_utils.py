from lib.utils.viz.load_data import data, global_titles, bar_source
import altair as alt
import numpy as np
import pandas as pd
global  data, global_titles, bar_source

def noise_plot(source=data['students']['kd']['noise'],
               title="Diferencia con ResNet101 contra ruido",
               y='delta_teacher',
               x='noise',
               color='feat_dist',
               shape='layer',
               student='ResNet18',
               xscale='linear',
               yscale='linear'):
    
        d=locals()
        ks=[i for i in d.keys() if i not in ['source','title','xscale','yscale','scale','bs'] ]     
        vals=[d[i] for  i in ks ]
        source = source.drop(columns=[i for i in source.columns if i not in vals+['student','noise']])

        sou=source[source['student']==student]
        
        #reduce data

        
        selection = alt.selection_multi(fields=[color], bind='legend')


        ytitle=global_titles[y] if yscale=='linear' else global_titles[y]+" [%s]"%yscale
        xtitle=global_titles[x] if xscale=='linear' else global_titles[x]+" [%s]"%xscale

        base = alt.Chart(sou,title=title).mark_line().encode(
            alt.Y(y, type='quantitative',title=ytitle,scale=alt.Scale(zero=False,base=10,type=yscale)),
            alt.X('noise:Q', type='quantitative',scale=alt.Scale(zero=False,base=10,type=xscale),title=xtitle),
            color=alt.Color('%s:N'%color, legend=alt.Legend(title=global_titles[color]), scale=alt.Scale(scheme='spectral')),
            
            strokeDash=alt.Stroke('%s:N'%shape, legend=alt.Legend(title=global_titles[shape])),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2))

        ).add_selection(
            selection
        ).properties(
            width=600,
            height=600
        )


        line = alt.Chart(sou).mark_line().encode(
            x=x,
            #y='mean(%s)'%y,
            
            y=alt.Y('mean(%s)'%y, type='quantitative',title=ytitle,scale=alt.Scale(zero=False,base=10,type=yscale)),
            color=alt.Color('%s:N'%color, legend=alt.Legend(title=global_titles[color]), scale=alt.Scale(scheme='spectral')),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
            shape=alt.Shape('%s:N'%color, legend=alt.Legend(title='media')),
            #opacity=alt.value(0.5)
        )
        
        
        point = alt.Chart(sou).mark_point(filled=True).encode(
            x=x,
            #y='mean(%s)'%y,
            
            y=alt.Y('mean(%s)'%y, type='quantitative',title=ytitle,scale=alt.Scale(zero=False,base=10,type=yscale)),
            color=alt.Color('%s:N'%color, legend=alt.Legend(title=global_titles[color]), scale=alt.Scale(scheme='spectral')),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
            shape=alt.Shape('%s:N'%color, legend=alt.Legend(title='media')),
            #opacity=alt.value(0.5)
        )
    

        band = alt.Chart(sou).mark_errorband(extent='ci').encode(
            x=alt.X('noise', title=xtitle),
            y=alt.Y('%s'%y, title=global_titles[y]),
            color=alt.Color('%s:N'%color, legend=alt.Legend(title=global_titles[color]), scale=alt.Scale(scheme='spectral')),
            opacity=alt.condition(selection, alt.value(0.4), alt.value(0.05))
        )

        w=base+ band + line + point
        w.properties(
            width=600,
            height=600
        )
        return w
def grafico_con_barra_xy_CE(source=data['students']['kd']['KD'],
                        title="Ratio entre exactitud en validacion y accuracy en entrenamiento.",
                        y="test/train",
                        x="temp",
                        color='student',
                        shape='log_dist',
                        bs=True,
                        fill=None,
                        xscale='log',
                        yscale='linear'):
    
    
            
        #reduce data
    d=locals()
    ks=[i for i in d.keys() if i not in ['source','title','xscale','yscale','scale','bs'] ]     
    vals=[d[i] for  i in ks ]
    source = source.drop(columns=[i for i in source.columns if i not in vals])
        
    ytitle=global_titles[y] if yscale=='linear' else global_titles[y]+" [%s]"%yscale
    xtitle=global_titles[x] if xscale=='linear' else global_titles[x]+" [%s]"%xscale
    encodings={'y':alt.Y(y, type='quantitative',title=ytitle,scale=alt.Scale(zero=False,base=10,type=yscale)),
            'x':alt.X(x, type='quantitative',title=xtitle,scale=alt.Scale(zero=False,base=10,type=xscale)),
            'color':alt.Color(color, legend=alt.Legend(title=global_titles[color])),
            'shape':alt.Shape(shape, legend=alt.Legend(title=global_titles[shape])),
            #size=50
              }
    
    if fill is not None:
        encodings["fill"]=alt.Fill(fill)

    chart=alt.Chart(source,title=title).mark_point(size=100).encode(**encodings).interactive()
    if bs:
        bar=alt.Chart(bar_source).mark_rule(opacity=0.5).encode(
                            y=y,
                            color=alt.Color('model', legend=alt.Legend(title="Modelo")),
                            stroke=alt.Stroke('model', legend=alt.Legend(title="Modelo en Cross Entropy")),
                            size=alt.value(2))
        d=bar+chart
        d.properties(width=600,height=600).configure_axis(titleFontSize=12).configure_title(fontSize=15)

        return d
    chart.properties(width=600,height=600).configure_axis(titleFontSize=12).configure_title(fontSize=15)
    return chart
    
def grafico_con_barra_y_CE(source=data['students']['kd']['KD'],
                        title="Diferencia de accuracy en entrenamiento con respecto a Resnet101.",
                        y="delta_teacher",
                        x="feat_dist",
                        fill='layer',
                        color='student',
                        shape='layer',
                        bs=True,
                        scale='linear'):
    
    
    if x=='dists':
        source=source.copy()
        source['dists']=source['feat_dist']+", "+source['log_dist']
    elif x=='feat,block':
        source=source.copy()
        source['feat,block']=[i+", "+str(j) for i,j in list(zip(source['feat_dist'],source['layer']))]
        
            
    #reduce data
    d=locals()
    ks=[i for i in d.keys() if i not in ['source','title','xscale','yscale','scale','bs'] ]      
    vals=[d[i] for  i in ks ]
    source = source.drop(columns=[i for i in source.columns if i not in vals])
        
    chart=alt.Chart(source,title=title).mark_point(size=100).encode(
           
            y=alt.Y(y, type='quantitative',scale=alt.Scale(zero=True,base=2,constant=1,type=scale),
                    title=global_titles[y] if scale=='linear' else global_titles[y]+" [%s]"%scale),
            x=alt.X(x, title=global_titles[x]),
            color=alt.Color(color, legend=alt.Legend(title=global_titles[color])),
            fill=alt.Fill('%s:O'%fill, legend=alt.Legend(title=global_titles[fill]), 
                          scale=alt.Scale(scheme='pastel1')),
            shape=alt.Shape("%s:O"%shape,legend=alt.Legend(title=global_titles[shape])),
            opacity=alt.value(0.5)
        )


    bar=alt.Chart(bar_source).mark_rule(opacity=0.5).encode(
                        y=y,
                        color=alt.Color('model', legend=alt.Legend(title="Modelo")),
                        stroke=alt.Stroke('model', legend=alt.Legend(title="Modelo en Cross Entropy")),
                        size=alt.value(2))

    if bs:
        bar=alt.Chart(bar_source).mark_rule(opacity=0.5).encode(
                            y=y,
                            color=alt.Color('model', legend=alt.Legend(title="Modelo")),
                            stroke=alt.Stroke('model', legend=alt.Legend(title="Modelo en Cross Entropy")),
                            size=alt.value(2))
        d=bar+chart
        d.properties(width=600,height=600).configure_axis(titleFontSize=12).configure_title(fontSize=15)

        return d
    
    chart.properties(width=600,height=600).configure_axis(titleFontSize=12).configure_title(fontSize=15)
    return chart
    

def grafico_con_columnas(source,
                         y='test_acc',
                         title="Exactitud en validacion desacoplado segun destilacion",
                         shape='student',
                         column='feat_dist',
                         color='layer',
                         fill=None
                        ):
    
            
    #reduce data
    d=locals()
    ks=[i for i in d.keys() if i not in ['source','title','xscale','yscale','scale','bs'] ]       
    vals=[d[i] for  i in ks ]
    source = source.drop(columns=[i for i in source.columns if i not in vals])
        
    encodings={"shape":alt.Shape("%s:O"%shape,legend=alt.Legend(title=global_titles[shape])),
        "y":alt.Y(y,title=global_titles[y]),
        "column":alt.Column('%s:O'%column,title=global_titles[column]),
        "x":alt.X('%s:N'%color,title=global_titles[color]),
        "color":alt.Color('%s:N'%color,legend=alt.Legend(title=global_titles[color])),
        "opacity":alt.value(0.5)}
    
    if fill is not None:
        encodings["fill"]=alt.Fill('%s:O'%fill, legend=alt.Legend(title=global_titles[fill]),scale=alt.Scale(scheme='pastel1'))
    d1=alt.Chart(source,title=title).mark_point(size=100).encode(**encodings
        ).configure_axis(titleFontSize=12,labelFontSize=12).configure_title(fontSize=15).interactive()
    return d1.properties(width=70,height=600)
