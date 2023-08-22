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
df_cl.to_excel('cluster/cluster_all.xlsx')
df_cl_cr.to_excel('cluster/cluster_all_strategy.xlsx')

#%%
# X_distance = gower.gower_matrix(df_t)
model = AgglomerativeClustering(distance_threshold=0,n_clusters=None)
# model = AgglomerativeClustering(distance_threshold=0,n_clusters=None)
model = model.fit(df_tt.values)
plot_dendrogram(model,truncate_mode='lastp',p=3)
#%%
df_cl.to_csv('cluster/cluster_60.csv')
df2.to_csv('cluster/strategy_all_cross.csv')