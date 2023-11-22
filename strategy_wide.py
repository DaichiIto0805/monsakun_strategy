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
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from sklearn.preprocessing import LabelEncoder
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
#%%
fname = 'strategy_wide/strategy_wide_5.csv'
fnum = re.sub(r"\D","",fname)
df = pd.read_csv(fname)
df2 = df.copy()
drop_col = ['InputID','day']
df = df.drop(drop_col,axis=1)
l = df.iloc[:,1:].values.tolist()
l = list(itertools.chain.from_iterable(l))
l = set(l)
data_dict = {}

for char in l:
    if char not in data_dict.keys():
        data_dict[char] = len(data_dict) #data_dictのサイズを数値ラベルとして付与

print(data_dict)
df = df.replace(data_dict)
#%%
clus3col = []
# clus2col = ['4_3_1','5_3_1']
clus2col = []
p=4

for j in range(len(df.columns)//15):
    df_ct = df.iloc[:,j*15:j*15+15]
    df_ct = df_ct.dropna()
    for i in range(len(df_ct.columns)//5):
        df_ctt = df_ct.iloc[:,i*5:i*5+5]
        if any(df_ctt.columns.values[0] == col for col in clus3col):
            p = 3
        elif any(df_ctt.columns.values[0] == co for co in clus2col):
            p = 2
        else:
            p = 2
        # X_distance = gower.gower_matrix(df_t)
        model = AgglomerativeClustering(distance_threshold=0,n_clusters=None)
        # model = AgglomerativeClustering(distance_threshold=0,n_clusters=None)
        model = model.fit(df_ctt.values)
        plot_dendrogram(model,truncate_mode='lastp',p=p)
        plt.title(df_ctt.columns.values[0]+'with'+str(p)+'cluster')
        plt.savefig('strategy_wide/'+fnum+'_'+df_ctt.columns.values[0]+'_with_'+str(p)+'_cluster_st_wide')
        plt.show(df_ctt.columns.values[0]+str(p)+'cluster')
        plt.clf()

        sse = []
        k_values = range(1, 11)

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df_ctt)
            sse.append(kmeans.inertia_)

        # SSEをプロット
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, sse, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('SSE')
        plt.title('Elbow Method For Optimal Number of Clusters'+fnum+'_'+df_ctt.columns.values[0])
        plt.savefig('strategy_wide/Elbow_'+fnum+'_'+df_ctt.columns.values[0]+'.png')
        plt.show()
#%%
def inverse_dict(d):
    return {v:k for k,v in d.items()}

#%%
k = 0
p = 3
df_cl = df2.loc[:,['InputID','day']]
df_cl_cr = df2.loc[:,['InputID','day']]
for j in range(len(df.columns)//15):
    df_t = df.iloc[:,j*15:j*15+15]
    df_t = df_t.dropna()
    for i in range(len(df_t.columns)//5):
        df_tt = df_t.iloc[:,i*5:i*5+5]
        df_tt2 = df_tt.replace(inverse_dict(data_dict))
        model = AgglomerativeClustering(n_clusters=p)
        # model = AgglomerativeClustering(distance_threshold=0,n_clusters=None)
        model = model.fit(df_tt.values)
        # plot_dendrogram(model,truncate_mode='lastp',p=3)
        df_tt['cluster'+str(k)]=model.labels_
        df_tt2['cluster'+str(k)]=model.labels_
        df_cl = pd.merge(df_cl,df_tt['cluster'+str(k)],how='left',left_index=True,right_index=True)
        df_cl_cr = pd.merge(df_cl_cr,df_tt2,how='left',left_index=True,right_index=True)
        print(k)
        k = k+1
fnum = re.sub(r"\D","",fname)
df_cl_cr.to_csv('strategy_wide/cluster_strategy_wide_'+fnum+'.csv')
#%%%%%%%%%%%%
print(df_cl_cr.index.values)