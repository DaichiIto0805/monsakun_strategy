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
import gower
# 階層クラスタリング
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import dendrogram
#%%
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1 # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)
# %%
df=pd.read_csv('strategy_all_cross/strategy_all_cross.csv',index_col=0)
# df = df[df['counts']==60]
df2 = df.copy()
df = df.drop('counts',axis=1)
# print(df)

df = df.replace({'F*_f':0,'F*_s':1,'r_f':2,'r_s':3,'s_f':4,'s_s':5,'n_f':6,'n_s':7})

# df_cl = df2['counts']

# %%
df_cl = df2['counts']
df_cl_cr = df2['counts']
k = 0
for j in range(len(df.columns)//15):
    df_t = df.iloc[:,j*15:j*15+15]
    df_t = df_t.dropna()
    for i in range(len(df_t.columns)//5):
        df_tt = df_t.iloc[:,i*5:i*5+5]
        df_tt2 = df_tt.replace({0:'F*_f',1:'F*_s',2:'r_f',3:'r_s',4:'s_f',5:'s_s',6:'n_f',7:'n_s'})
        model = AgglomerativeClustering(n_clusters=3)
        # model = AgglomerativeClustering(distance_threshold=0,n_clusters=None)
        model = model.fit(df_tt.values)
        # plot_dendrogram(model,truncate_mode='lastp',p=3)
        df_tt['cluster'+str(k)]=model.labels_
        df_tt2['cluster'+str(k)]=model.labels_
        df_cl = pd.merge(df_cl,df_tt['cluster'+str(k)],how='left',left_index=True,right_index=True)
        df_cl_cr = pd.merge(df_cl_cr,df_tt2,how='left',left_index=True,right_index=True)
        print(k)
        k = k+1

#%%
df_cl.to_csv('cluster/cluster_60.csv')
df2.to_csv('cluster/strategy_all_cross.csv')
#%%
df_cl.to_excel('cluster/cluster_all.xlsx')
df_cl_cr.to_excel('cluster/cluster_all_strategy.xlsx')

#%%
#デンドログラム作成
p = 4
for j in range(len(df.columns)//15):
    df_t = df.iloc[:,j*15:j*15+15]
    df_t = df_t.dropna()
    for i in range(len(df_t.columns)//5):
        df_tt = df_t.iloc[:,i*5:i*5+5]
        # X_distance = gower.gower_matrix(df_t)
        model = AgglomerativeClustering(distance_threshold=0,n_clusters=None)
        # model = AgglomerativeClustering(distance_threshold=0,n_clusters=None)
        model = model.fit(df_tt.values)
        plot_dendrogram(model,truncate_mode='lastp',p=p)
        plt.title(df_tt.columns.values[0]+'with'+str(p)+'cluster')
        plt.savefig('cluster/'+df_tt.columns.values[0]+'_with_'+str(p)+'_cluster')
        plt.show(df_tt.columns.values[0]+str(p)+'cluster')
        plt.clf()
#%%

df3 = df2.drop('counts',axis=1)
colnames = df3.columns.values
# print(df3.iloc[:,0])
for x in range(len(df3.columns)):
    df3 = pd.concat([df3,df3.iloc[:,x].str.split('_',expand= True)],axis=1).drop(colnames[x],axis=1)
    df3.rename(columns={0:colnames[x]+'_0',1:colnames[x]+'_1'},inplace=True)
print(df3)
#%%
print(df_tt.columns.values[0])



#%%
#正解率を利用したクラスター作成
df_corate = pd.read_csv('cluster/strategy_all_cross_with_correct_rate.csv',index_col=0)
df_corate2 = df_corate.copy()
df_corate = df_corate.drop('counts',axis=1)
df_corate = df_corate.replace('(.*)_(.*)',r'\1',regex=True)
df_corate = df_corate.replace({'F*':0,'s':1,'r':2,'n':3})
print(df_corate.iloc[:,0:18])
#%%
#デンドログラム作成（デンドログラムを見てクラスター数を設定）
clus3col = ['2_1_1','3_3_1','4_2_1','5_1_1']
clus2col = ['4_3_1','5_3_1']
for j in range(len(df_corate.columns)//18):
    df_ct = df_corate.iloc[:,j*18:j*18+18]
    df_ct = df_ct.dropna()
    for i in range(len(df_ct.columns)//6):
        df_ctt = df_ct.iloc[:,i*6:i*6+6]
        if any(df_ctt.columns.values[0] == col for col in clus3col):
            p = 3
        elif any(df_ctt.columns.values[0] == co for co in clus2col):
            p = 2
        else:
            p = 4
        # X_distance = gower.gower_matrix(df_t)
        model = AgglomerativeClustering(distance_threshold=0,n_clusters=None)
        # model = AgglomerativeClustering(distance_threshold=0,n_clusters=None)
        model = model.fit(df_ctt.values)
        plot_dendrogram(model,truncate_mode='lastp',p=p)
        plt.title(df_ctt.columns.values[0]+'with'+str(p)+'cluster')
        plt.savefig('cluster_with_corate/'+df_ctt.columns.values[0]+'_with_'+str(p)+'_cluster_corate')
        plt.show(df_ctt.columns.values[0]+str(p)+'cluster')
        plt.clf()
#%%
#クラスター作成
df_cor_cl = df_corate2['counts']
df_cor_cl_cr = df_corate2['counts']
k = 0
for j in range(len(df_corate.columns)//18):
    df_ct = df_corate.iloc[:,j*18:j*18+18]
    df_ct = df_ct.dropna()
    for i in range(len(df_ct.columns)//6):
        df_ctt = df_ct.iloc[:,i*6:i*6+6]
        if any(df_ctt.columns.values[0] == col for col in clus3col):
            p = 3
        elif any(df_ctt.columns.values[0] == co for co in clus2col):
            p = 2
        else:
            p = 4
        df_ctt2 = df_ctt.iloc[:,0:5].replace({0:'F*',1:'s',2:'r',3:'n'})
        df_ctt2.insert(5,'correct_rate'+str(k),df_ctt.iloc[:,5])
        model = AgglomerativeClustering(n_clusters=p)
        # model = AgglomerativeClustering(distance_threshold=0,n_clusters=None)
        model = model.fit(df_ctt.values)
        # plot_dendrogram(model,truncate_mode='lastp',p=3)
        df_ctt['cluster'+str(k)+'('+str(p)+')']=model.labels_
        df_ctt2['cluster'+str(k)+'('+str(p)+')']=model.labels_
        df_cor_cl = pd.merge(df_cor_cl,df_ctt['cluster'+str(k)+'('+str(p)+')'],how='left',left_index=True,right_index=True)
        df_cor_cl_cr = pd.merge(df_cor_cl_cr,df_ctt2,how='left',left_index=True,right_index=True)
        print(k)
        k = k+1
df_cor_cl.to_csv('cluster_with_corate/cluster_all_corate.csv')
df_cor_cl_cr.to_excel('cluster_with_corate/cluster_strategy_corate.xlsx')
#%%----------------------------------------------------------------
#クラスターのクラスター分析
#デンドログラム作成
p = 4
df_cl_cl = pd.read_csv('cluster_with_corate/cluster_all_corate.csv',index_col=0)
df_cl_cl = df_cl_cl.drop('counts',axis=1)
cl3cl = ['cluster0(3)','cluster3(4)']
for j in range(len(df_cl_cl.columns)//3):
    df_clclct = df_cl_cl.iloc[:,j*3:j*3+3]
    df_clclct = df_clclct.dropna()
    if any(df_clclct.columns.values[0]==cl for cl in cl3cl):
        p = 3
    else:
        p = 4
    model = AgglomerativeClustering(distance_threshold=0,n_clusters=None)
    # model = AgglomerativeClustering(distance_threshold=0,n_clusters=None)
    model = model.fit(df_clclct.values)
    plot_dendrogram(model,truncate_mode='lastp',p=p)
    plt.title(df_clclct.columns.values[0]+'with'+str(p)+'cluster')
    plt.savefig('cluster_cluster/'+df_clclct.columns.values[0]+'_with_'+str(p)+'_cluster')
    plt.show(df_clclct.columns.values[0]+str(p)+'cluster')
    plt.clf()
#%%
#クラスター作成
df_clclct_cl = pd.DataFrame(index=df_cl_cl.index,columns=[])

for j in range(len(df_cl_cl.columns)//3):
    df_clclct = df_cl_cl.iloc[:,j*3:j*3+3]
    df_clclct = df_clclct.dropna()
    if any(df_clclct.columns.values[0]==cl for cl in cl3cl):
        p = 3
    else:
        p = 4
    model = AgglomerativeClustering(n_clusters=p)
    # model = AgglomerativeClustering(distance_threshold=0,n_clusters=None)
    model = model.fit(df_clclct.values)
    df_clclct['clcl'+str(j)+'('+str(p)+')']=model.labels_
    df_clclct_cl = pd.merge(df_clclct_cl,df_clclct,how='left',left_index=True,right_index=True)

df_clclct_cl.to_excel('cluster_cluster/cluster_cluster.xlsx')