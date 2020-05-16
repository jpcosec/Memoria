from lib.utils.viz.load_data import data, global_titles, bar_source
import numpy as np
import pandas as pd
global  data, global_titles, bar_source

def get_top(source, by=['delta_student'],
            n_cases=5,add_fields=[],drop_fields=[]):
    
    
    drop = ['test_loss', 'test/train', 'train_acc', 'train_loss', 'transform', 'dataset', 'noise', 'loss_conv',
            'acc_conv', 'max', 't','test_acc']+drop_fields
    drop =[i for i in drop if i in source.columns]
    drop = [i for i in drop if not i in add_fields]
    
    top = source.sort_values(by=by, ascending=False).drop(columns=drop).head(n_cases)
    
    return top

def top_latex(source,titles,n_cases=5,keep=[],drop=[]):
    keep_cols=keep+['delta_student','delta_teacher']
    s=source.copy()
    s_cols=[i for i in s.columns if i not in keep_cols]
    s['D. logits']= ["\\footnotesize "+", ".join([i,str(j)]) for i,j in list(zip(s['log_dist'],s['temp']))]
    s['D. Conv.']= ["\\footnotesize "+", ".join([i,str(j)]) if i !='-' else "" for i,j in list(zip(s['feat_dist'],s['layer'])) ]
    s['Modelo']= ["\\footnotesize "+i for i in s['student']]
    
    idx=['Modelo','D. logits','D. Conv.']+keep_cols
    ixd=[i for i in idx if i not in drop]
    s=s.drop(columns=s_cols+drop).reindex(idx,axis='columns')
    j=['delta_student','delta_teacher']
    t=[s.sort_values(by=j[f], ascending=False).drop(columns=[j[-1-f] ]+drop).head(n_cases).copy() for f in [0,1]]
    
    s=["\n".join([i+" \hline" for i in table.to_latex(index=False).split("\n")[2:-3]]) for table in t]

    table="""
    \\begin{minipage}[b]{.40\\textwidth}

    \\small
    \\begin{tabular}{|l|l|l|l|r|r|r|}
        \\hline
        \\rowcolor[HTML]{9B9B9B} 
        %s
        \\end{tabular}
       \\captionof{table}{%s}
       \\label{tab:forpol}
    \\end{minipage}\\qquad
    \\begin{minipage}[b]{.40\\textwidth}
    \\hspace{1.5cm}
    \\small
       \\begin{tabular}{|l|l|l|l|r|r|r|}
        \\hline
        \\rowcolor[HTML]{9B9B9B} 
        %s
        \\end{tabular}
       \\captionof{table}{%s}
       \\label{tab:revpol}
    \\end{minipage}

    """%(s[0],titles[0],s[1],titles[1])
      
    return table.replace("textbackslash ","").replace("\midrule \hline","").replace(" delta\_teacher","$\Delta T$").replace(" delta\_student","$\Delta S$")

