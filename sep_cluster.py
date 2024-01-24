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
#%%%%
fnames = glob.glob('strategy_wide/strategy_wide_?_sep.csv')
for f in fnames:
    fnum = re.sub(r'\D', '', f)
    print(f)
    dfsep = pd.read_csv(f,index_col=0)
    dfsep = dfsep.sort_index()
    dfsep2 = dfsep.loc[:,['InputID','day']]
    dfsep.drop(columns=['InputID','day'],inplace=True)
    selected_col = [col for col in dfsep.columns if 'cluster' in col]
    dfsep.drop(columns =selected_col,inplace=True)
    for i in range(len(dfsep.columns)//15):
        print(i)
        df = dfsep.iloc[:,i*15:(i*15)+15]
        df = df.replace({'x':0,'F':1,'R':1,'S':1})
        sum_F = df.filter(regex='F$').sum(axis=1)
        sum_F.rename(str(i+1)+'_F',inplace=True)
        sum_S = df.filter(regex='S$').sum(axis=1)
        sum_S.rename(str(i+1)+'_S',inplace=True)
        sum_R = df.filter(regex='R$').sum(axis=1)
        sum_R.rename(str(i+1)+'_R',inplace=True)

        dfsep2 = pd.concat([dfsep2,sum_F,sum_S,sum_R],axis=1)
    dfsep2.to_csv('strategy_sep/strategy_sep_count_'+str(fnum)+'.csv')
#%%%%
from chi import residual_analysis

p = 4
l = glob.glob('strategy_sep/strategy_sep_count_*')
for i in l:
    print(i)
    dfcl = pd.read_csv(i,index_col=0)
    fnum = re.sub(r'\D', '', i)
    dfcl2 = dfcl.loc[:,['InputID','day']]
    dfcl.drop(columns=['InputID','day'],inplace=True)
    for j in range(len(dfcl.columns)//3):
        dfclel = dfcl.iloc[:,j*3:(j*3)+3]
        model = AgglomerativeClustering(n_clusters=p)
        model = model.fit(dfclel.values)
        dfclel.loc[:,'cluster'+str(j)] = model.labels_
        dfcl2 = pd.concat([dfcl2,dfclel],axis=1)
        dfred = dfclel.groupby('cluster'+str(j)).sum()
        try :
            pairs = residual_analysis(table=dfred,p_value=0.05)
        except ValueError as e :
            print(e)
        # print(pairs)

        for k in range(1,len(dfred.columns)+1):
            dfred.insert(k*2-1,'re'+dfred.columns.values[(k-1)*2],'')

        for y in dfred.columns:
            for x in (dfred.index):
                for pair in pairs:
                    if pair[0] == x and pair[1] == y:
                            dfred.iloc[dfred.index.get_loc(x),dfred.columns.get_loc(y)+1] = pair[2]

        dfred.to_excel('strategy_sep/residual_analysis/residual_analysis_'+str(fnum)+'_lv_'+str(j+1)+'.xlsx')

    dfcl2.to_csv('./strategy_sep/cluster_strategy_sep_'+str(fnum)+'.csv')
#%%
import seaborn as sns
#%%
fnames = glob.glob('strategy_sep/cluster_strategy_sep_5.csv')
for f in fnames:
    dfcl2 = pd.read_csv(f,index_col=0)
    fnum = re.sub(r'\D', '', f)
    df1 = pd.read_csv('monsakun_log_0'+str(fnum)+'.csv')
    df1 = df1.query('ope1 in ["CHECK"]')
    df1 = df1.loc[:,['InputID','check','lv','asg']]
    df1_lv1 = df1.query('lv in [1]')
    df1_lv2 = df1.query('lv in [2]')
    df1_lv3 = df1.query('lv in [3]')

    dfcl2 = dfcl2.sort_index()
    df1_lv1 = pd.merge(df1_lv1,dfcl2.loc[:,['InputID','cluster0']],on='InputID')
    df1_lv2 = pd.merge(df1_lv2,dfcl2.loc[:,['InputID','cluster1']],on='InputID')
    df1_lv3 = pd.merge(df1_lv3,dfcl2.loc[:,['InputID','cluster2']],on='InputID')
    df1_lv1.drop(columns=['InputID'],inplace=True)
    df1_lv1.rename(columns={'cluster0':'cluster'},inplace=True)
    df1_lv1 = df1_lv1.sort_values('cluster')
    df1_lv2.drop(columns=['InputID'],inplace=True)
    df1_lv2.rename(columns={'cluster1':'cluster'},inplace=True)
    df1_lv3.drop(columns=['InputID'],inplace=True)
    df1_lv3.rename(columns={'cluster2':'cluster'},inplace=True)
    dflv = pd.concat([df1_lv1,df1_lv2,df1_lv3])
    dflv = dflv.sort_values(by=['lv','cluster'])
    dflv = dflv.reindex(columns=['lv','cluster','check'])
    dflv_mean = dflv.groupby(['lv','cluster'])['check'].mean()
    sns.boxplot(x='lv',y='check',hue='cluster',data=df1_lv3)
    plt.title('cluster_check'+str(fnum))
    plt.savefig('strategy_sep/imgs/cluster_check_'+fnum+'.png')
    plt.show()
    print(dflv_mean)
    plt.clf()

    sns.lineplot(x='asg',y='check',hue='cluster',data=df1_lv3)
    # df1_lv3.plot()
    plt.show()
    plt.style.use('default')
    plt.clf()
# %%
import scikit_posthocs as sp


# display(sp.posthoc_dscf(df1_lv3.query('cluster in [1,3]'), val_col='check', group_col='cluster'))
display(sp.posthoc_dscf(df1_lv3, val_col='check', group_col='cluster'))
display(df1_lv3.groupby('cluster').describe())
# %%
