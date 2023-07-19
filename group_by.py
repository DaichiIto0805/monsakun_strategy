#%%
from sklearn import datasets, preprocessing
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import glob
import re

from pyclustering.cluster import gmeans, xmeans
import itertools
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
import platform
from scipy import stats
import itertools

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.ticker import MaxNLocator

sns.set(font='IPAexGothic')
#%%
files= glob.glob('log_check/monsakun_log_check_r_s??.csv')
print(files)
# モンサクンのチェックと戦略のファイル
for num,file in enumerate(files):
    fnum = re.sub(r"\D","",file)
    fnum = fnum[:2]
    print("fnum=",fnum)

    df = pd.read_csv(file,index_col=0)
    df = df[df['jdg']=='s']
    # 重複削除
    df = df.drop_duplicates(subset=['InputID','q'])
    # display(df)
    vc = df['InputID'].value_counts()
    vc = vc.rename('counts')
    df = pd.merge(df,vc,left_on='InputID',right_index=True)
    # vc = pd.unique(df['counts'])
    solve_count = pd.unique(df['counts'])

    query_str = "story_st == 1 and formula_st == 1"
    df_subset = df.query(query_str)
    df.loc[df_subset.index,"story_st"]=0
    query_str = "relation_st == 1 and formula_st == 1"
    df_subset = df.query(query_str)
    df.loc[df_subset.index,"relation_st"]=0
    df.relation_st[df.relation_st==1]='r'
    df.relation_st[df.relation_st==0]=''
    df.story_st[df.story_st==1]='s'
    df.story_st[df.story_st==0]=''
    df.formula_st[df.formula_st==1]='F*'
    df.formula_st[df.formula_st==0]=''
    df['strategy']=df['relation_st']+df['story_st']+df['formula_st']
    df.strategy[df.strategy=='']='n'

    df.to_csv('monsakun_log_per_solve_counts/monsakun_log_per_solve_counts'+str(fnum)+'.csv')

    for k in solve_count:
        df1 = df[df['counts']==k]
        df2 = pd.crosstab(df1['strategy'],df1['q'],margins=True)
        df2.to_csv('strategy_per_solve_count/strategy_per_solve_count_'+str(fnum)+'_'+str(k)+'.csv')

        #積み上げ棒グラフ描画
        df2 = pd.crosstab(df1['strategy'],df1['q'])
        #積み上げ棒グラフ描画
        ig, ax = plt.subplots(figsize=(10,8))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        storategy_list = ['F*','n','r','s']
        colorlist = mcolors.TABLEAU_COLORS.keys()
        dict_colorlist = dict(zip(storategy_list,colorlist))

        for i in range(len(df2)):
            ax.bar(df2.columns, df2.iloc[i], bottom=df2.iloc[:i].sum(),color=dict_colorlist[df2.index[i]])
            for j in range(len(df2.columns)):
                plt.text(x=j+1,
                            y=df2.iloc[:i, j].sum() + (df2.iloc[i, j] / 2),
                            s=df2.iloc[i, j],
                            ha='center',
                            va='bottom'
                        )
        ax.set(xlabel='q', ylabel='strategy')
        ax.legend(df2.index)
        plt.title('first_check_strategy_lv_'+str(fnum)+'_'+str(k))
        plt.savefig('strategy_per_solve_graphs/strategy_per_solve'+str(fnum)+'_'+str(k)+'.png')
        plt.show()
