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

from sklearn.preprocessing import StandardScaler
# K-Prototypeクラスタリング
from kmodes.kprototypes import KPrototypes
# Gower距離
# import gower
# 階層クラスタリング
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import dendrogram
#%%
df1 = pd.read_csv('monsakun_log_04.csv')
df1 = df1.query('ope1 in ["CHECK"]')
df1 = df1.loc[:,['InputID','check','lv','asg']]
df1_lv1 = df1.query('lv in [1]')
df1_lv2 = df1.query('lv in [2]')
df1_lv3 = df1.query('lv in [3]')
df2 = pd.read_csv('strategy_wide/cluster_strategy_wide_4_meaning.csv',index_col=0)
df2 = df2.sort_index()
df1_lv1 = pd.merge(df1_lv1,df2.loc[:,['InputID','cluster0']],on='InputID')
df1_lv1 = df1_lv1.loc[:,['cluster0','check','asg']].groupby(['cluster0','asg'],as_index=False).mean()
df1_lv2 = pd.merge(df1_lv2,df2.loc[:,['InputID','cluster1']],on='InputID')
df1_lv2 = df1_lv2.loc[:,['cluster1','check','asg']].groupby(['cluster1','asg'],as_index=False).mean()
df1_lv3 = pd.merge(df1_lv3,df2.loc[:,['InputID','cluster2']],on='InputID')
df1_lv3 = df1_lv3.loc[:,['cluster2','check','asg']].groupby(['cluster2','asg'],as_index=False).mean()
lv1_piv = df1_lv1.pivot(index='cluster0',columns='asg',values='check')
lv2_piv = df1_lv2.pivot(index='cluster1',columns='asg',values='check')
lv3_piv = df1_lv3.pivot(index='cluster2',columns='asg',values='check')
# %%
l = glob.glob('strategy_hist_sum/residual_analysis_4_lv_?.xlsx')
for x,j in enumerate([df1_lv1,df1_lv2,df1_lv3]):

    print(l[x])
    df_l = pd.read_excel(l[x],index_col=0)


    j = j[j.index.isin(df_l.index)]
    print(j.index)


    plt.bar(j.index,j['check'])
    for i in j.index:
        plt.text(x=i,y=j['check'][i],s=j['check'][i].round(2),ha='center')
    plt.title(str(j.index.name))
    plt.show()
# %%
from chi import residual_analysis
for x,i in enumerate([lv1_piv,lv2_piv,lv3_piv]):
    fnum = re.sub(r'\D', '', l[x])
    print(l[x])
    df_l = pd.read_excel(l[x],index_col=0)

    i = i[i.index.isin(df_l.index)]
    y_col = i.index.values
    i2 = i.T
    i2= i2.reset_index()
    i2.plot(x="asg",y=y_col,kind='bar')
    # i2.plot(x="asg",y=y_col)
    plt.title(fnum[0]+'_'+fnum[1])
    plt.grid(axis='y',linestyle='dotted')
    plt.savefig('strategy_hist_sum/check_mean/check_mean_'+fnum[0]+'_'+fnum[1]+'.png')
    plt.show()
    plt.clf()
    plt.close()

    i3 = i
    pairs = residual_analysis(table=i3,p_value=0.05)
    print(pairs)

    for j in range(1,len(i3.columns)+1):
        i3.insert(j*2-1,'re'+str(i3.columns.values[(j-1)*2]),'')

    for y in i3.columns:
        for x in (i3.index):
            for p in pairs:
                if p[0] == x and p[1] == y:
                    i3.iloc[i3.index.get_loc(x),i3.columns.get_loc(y)+1] = p[2]

    i3.index.name = str(fnum[0]) + '_' + str(fnum[1])

    i3.to_excel('strategy_hist_sum/residual_analysis_check_mean/residual_analysis_check_mean_'+fnum[0]+'_'+fnum[1]+'.xlsx')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%