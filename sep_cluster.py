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
import scikit_posthocs as sp
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
    dfcl = pd.read_csv(i,index_col=0)
    fnum = re.sub(r'\D', '', i)
    dfcl2 = dfcl.loc[:,['InputID','day']]
    dfcl.drop(columns=['InputID','day'],inplace=True)
    for j in range(len(dfcl.columns)//3):
        dfclel = dfcl.iloc[:,j*3:(j*3)+3]
        model = AgglomerativeClustering(n_clusters=p)
        model = model.fit(dfclel.values)
        # dfclel.loc[:,'cluster'+str(j)] = model.labels_
        dfclel = dfclel.assign(**{"cluster"+str(j):model.labels_})
        dfcl2 = pd.concat([dfcl2,dfclel],axis=1)
        dfred = dfclel.groupby('cluster'+str(j)).sum()
        dfratio = dfclel.groupby('cluster'+str(j))[[str(j+1)+'_F',str(j+1)+'_S',str(j+1)+'_R']].apply(lambda x: x.sum() / x.count())
        for inde,t in enumerate([dfred,dfratio]):
            try :
                print(i,j+1)
                pairs = residual_analysis(table=t,p_value=0.05)
            except ValueError as e :
                print(e)
            if inde == 1:
                t.to_csv('strategy_sep/strategy_sep_ratio/strategy_sep_ratio_'+str(fnum)+'_'+str(j+1)+'.csv')
            for k in range(1,len(t.columns)+1):
                t.insert(k*2-1,'re'+t.columns.values[(k-1)*2],'')
            if pairs != 0:
                for y in t.columns:
                    for x in (t.index):
                        for pair in pairs:
                            if pair[0] == x and pair[1] == y:
                                    t.iloc[t.index.get_loc(x),t.columns.get_loc(y)+1] = pair[2]
            if inde ==0 :
                t.to_excel('strategy_sep/residual_analysis/residual_analysis_sum_'+str(fnum)+'_lv_'+str(j+1)+'.xlsx')
            else:
                t.to_excel('strategy_sep/residual_analysis/residual_analysis_ratio_'+str(fnum)+'_lv_'+str(j+1)+'.xlsx')

    dfcl2.to_csv('./strategy_sep/cluster_strategy_sep_'+str(fnum)+'.csv')
#%%
import seaborn as sns
#%%
fnames = glob.glob('strategy_sep/cluster_strategy_sep_*.csv')
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
    plt.clf()

    plt.title('cluster_check'+str(fnum))
    xlabels = df1_lv3.loc[:,'asg'].unique()
    df1_lv3_mean = df1_lv3.groupby(['cluster','asg']).mean()
    df1_lv3_mean = df1_lv3_mean.reset_index()
    # sns.lineplot(x='asg',y='check',hue='cluster',data=df1_lv3_mean)
    df1_lv3_mean = df1_lv3_mean.loc[:,['asg','cluster','check']]
    df1_lv3_mean = df1_lv3_mean.pivot(index= 'asg',columns = 'cluster',values = 'check')
    plt.xticks(xlabels,xlabels)
    df1_lv3_mean.plot()
    plt.savefig('strategy_sep/imgs/cluster_check_bar_'+fnum+'.png')
    plt.show()
    plt.clf()

#!TODO分散分析
    dfsp = sp.posthoc_dscf(df1_lv3, val_col='check', group_col='cluster')
    dfsp.to_excel('strategy_sep/posthoc_dscf/posthoc_dscf'+str(fnum)+'.xlsx')
    print(df1_lv3.groupby('cluster').describe())
# %%
lists = glob.glob('strategy_sep/cluster_strategy_sep_?.csv')
for l in lists:
    fnum = re.sub(r'\D', '', l)
    print(l)
    dfs = pd.read_csv(l,index_col=0)
    dfs = dfs.loc[:,['InputID','cluster0','cluster1','cluster2']]
    dfs1 = dfs.loc[:,['InputID','cluster0']]
    dfs1['lv'] = 1
    dfs1 = dfs1.rename(columns={'cluster0':'cluster'})
    dfs2 = dfs.loc[:,['InputID','cluster1']]
    dfs2['lv'] = 2
    dfs2 = dfs2.rename(columns={'cluster1':'cluster'})
    dfs3 = dfs.loc[:,['InputID','cluster2']]
    dfs3['lv'] = 3
    dfs3 = dfs3.rename(columns={'cluster2':'cluster'})

    dfs = pd.concat([dfs1,dfs2,dfs3],axis=0)

    df7 = dfs.pivot_table(index='InputID',columns='lv',values='cluster')
    df7 = df7.sort_values(by=[1,2,3])
    df7[1]=df7[1].astype(pd.Int64Dtype(),errors='ignore')
    df7[2]=df7[2].astype(pd.Int64Dtype(),errors='ignore')
    df7[3]=df7[3].astype(pd.Int64Dtype(),errors='ignore')
    df7[1]='1_'+df7[1].astype(str)
    df7[2]='2_'+df7[2].astype(str)
    df7[3]='3_'+df7[3].astype(str)
    df7 = df7.stack()

    df8 = df7.reset_index()
    df8 = df8[['InputID',0]]
    df8['to']=df8[0].shift(-1)[df8['InputID']==df8['InputID'].shift(-1)].dropna()
    df8 = df8.rename(columns={0:'from'}).drop(columns=['InputID'])
    df8 = df8.dropna()
    df8 = df8.groupby(['from','to']).size().reset_index(name='value')

    src,tgt,val = df8['from'].values,df8['to'].values,df8['value'].values

    lbl = {e:i for i,e in enumerate(sorted(set(src) | set(tgt)))}

    src = [lbl[e] for e in src]
    tgt = [lbl[e] for e in tgt]
    lbl = list(lbl.keys()) # 店名＝ノード名

    # 描画
    fig = go.Figure([go.Sankey(
        node = dict( pad = 15, thickness = 20,
            line = dict(color = "black", width = 1),
            label = lbl, color = "blue" ),
        link = dict( source = src, target = tgt, value = val))])
    fig.show()

    fig.write_image('strategy_sep/sankey/sankey_'+str(fnum)+'.png')
#%%
ls = glob.glob('strategy_sep\cluster_strategy_sep_?.csv')
for l in ls:
    fnum = re.sub(r'\D', '', l)
    df = pd.read_csv(l,index_col=0)
    df = df.loc[:,['InputID','3_F','3_S','3_R','cluster2']]
    df = df.melt(id_vars=['InputID','cluster2'], var_name='st', value_name='value')
    df = df.drop('InputID',axis=1)
    sns.boxplot(x='cluster2',y='value',hue='st',data=df)
    plt.title('day'+str(fnum)+'_lv3')
    plt.savefig('strategy_sep\startegy_box/strategy_box_'+str(fnum)+'.png')
    plt.show()
    plt.clf()