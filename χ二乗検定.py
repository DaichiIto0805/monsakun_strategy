# %%
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

sns.set(font='IPAexGothic')

# %%
import os
dir = ['whole_level_graphs','strategy_lv']
for i in range(len(dir)):
    if not os.path.exists(dir[i]):
        os.mkdir(dir[i])

# %%
from matplotlib.ticker import MaxNLocator
files = glob.glob('log_check/monsakun_log_check_r_s??.csv')
for file1 in files:
    fnum1 = re.sub(r"\D","",file1)
    df1 = pd.read_csv('InputID_q_drop_'+str(fnum1)+'.csv',index_col=0)
    df2 = pd.read_csv(file1,index_col=0)
    df2 = df2[df2['check']==1]
    df3 = pd.merge(df2,df1['InputID'],on=['InputID'],how='inner')
    df4 = pd.crosstab([df3['relation_st'],df3['story_st'],df3['formula_st']],df3['lv'])

    fig, ax = plt.subplots(figsize=(10,8))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))



    #積み上げ棒グラフ描画
    st_list = [(0,0,0), (0,0,1),(0,1,0), (0,1,1), (1,0,0), (1,0,1)]
    colorlist = mcolors.TABLEAU_COLORS.keys()

    dict_colorlist = dict(zip(st_list,colorlist))

    for i in range(len(df4)):
        ax.bar(df4.columns, df4.iloc[i], bottom=df4.iloc[:i].sum(),color=dict_colorlist[df4.index[i]])
        for j in range(len(df4.columns)):
            plt.text(x=j+1, 
                        y=df4.iloc[:i, j].sum() + (df4.iloc[i, j] / 2), 
                        s=df4.iloc[i, j], 
                        ha='center', 
                        va='bottom'
                    )
    ax.set(xlabel='q', ylabel='strategy')
    ax.legend(df4.index)
    plt.title('first_check_whole_lv_'+str(fnum1))
    plt.savefig('whole_level_graphs/r_s_t_first_check_whole_lv'+str(fnum1)+'.png')
    plt.show()
# %%
files = glob.glob('log_check/monsakun_log_check_r_s??.csv')
for file1 in files:
    fnum1 = re.sub(r"\D","",file1)
    df1 = pd.read_csv('InputID_q_drop_'+str(fnum1)+'.csv',index_col=0)
    df2 = pd.read_csv(file1,index_col=0)
    df2 = df2[df2['check']==1]
    df3 = pd.merge(df2,df1['InputID'],on=['InputID'],how='inner')

    #ゴリ押しで整形（戦略のワンホットベクトルを直す）
    query_str = "story_st == 1 and formula_st == 1"
    df3_subset = df3.query(query_str)
    df3.loc[df3_subset.index,"story_st"]=0
    query_str = "relation_st == 1 and formula_st == 1"
    df3_subset = df3.query(query_str)
    df3.loc[df3_subset.index,"relation_st"]=0
    df3.relation_st[df3.relation_st==1]='r'
    df3.relation_st[df3.relation_st==0]=''
    df3.story_st[df3.story_st==1]='s'
    df3.story_st[df3.story_st==0]=''
    df3.formula_st[df3.formula_st==1]='F*'
    df3.formula_st[df3.formula_st==0]=''
    df3['strategy']=df3['relation_st']+df3['story_st']+df3['formula_st']
    df3.strategy[df3.strategy=='']='n'

    df4 = pd.crosstab(df3['strategy'],df3['lv'])

    ig, ax = plt.subplots(figsize=(10,8))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))



    #積み上げ棒グラフ描画

    for i in range(len(df4)):
        ax.bar(df4.columns, df4.iloc[i], bottom=df4.iloc[:i].sum())
        for j in range(len(df4.columns)):
            plt.text(x=j+1, 
                        y=df4.iloc[:i, j].sum() + (df4.iloc[i, j] / 2), 
                        s=df4.iloc[i, j], 
                        ha='center', 
                        va='bottom'
                    )
    ax.set(xlabel='q', ylabel='strategy')
    ax.legend(df4.index)
    plt.title('first_check_whole_lv_'+str(fnum1))
    plt.savefig('whole_level_graphs/r_s_t_first_check_whole_lv_2_'+str(fnum1)+'.png')
    plt.show()