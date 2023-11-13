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

#%%
df = pd.read_csv('strategy_all_cross/strategy_all_cross.csv',index_col=0)
df2 = df.copy()
df = df.drop('counts',axis=1)
df = df.replace('(.*)_(.*)',r'\1',regex=True)
df = df.replace({'F*':0,'s':1,'r':2,'n':3})
#%%
clus3col = ['4_2_1','5_1_1']
# clus2col = ['4_3_1','5_3_1']
clus2col = ['3_1_1']
# for j in range(len(df.columns)//15):
#     df_ct = df.iloc[:,j*15:j*15+15]
#     df_ct = df_ct.dropna()
#     for i in range(len(df_ct.columns)//5):
#         df_tt = df_ct.iloc[:,i*5:i*5+5]
#         if any(df_tt.columns.values[0] == col for col in clus3col):
#             p = 3
#         elif any(df_tt.columns.values[0] == co for co in clus2col):
#             p = 2
#         else:
#             p = 4
#         # X_distance = gower.gower_matrix(df_t)
#         model = AgglomerativeClustering(distance_threshold=0,n_clusters=None)
#         # model = AgglomerativeClustering(distance_threshold=0,n_clusters=None)
#         model = model.fit(df_tt.values)
#         plot_dendrogram(model,truncate_mode='lastp',p=p)
#         plt.title(df_tt.columns.values[0]+'with'+str(p)+'cluster')
#         plt.savefig('cluster_with_corate/'+df_tt.columns.values[0]+'_with_'+str(p)+'_cluster_corate')
#         plt.show(df_tt.columns.values[0]+str(p)+'cluster')
#         plt.clf()


#%%----------------------------------------------------------------
df_cl = df2['counts']
df_cl_cr = df2['counts']
#%%
k = 0
for j in range(len(df.columns)//15):
    df_t = df.iloc[:,j*15:j*15+15]
    df_t = df_t.dropna()
    for i in range(len(df_t.columns)//5):
        df_tt = df_t.iloc[:,i*5:i*5+5]
        if any(df_tt.columns.values[0] == col for col in clus3col):
            p = 3
        elif any(df_tt.columns.values[0] == co for co in clus2col):
            p = 2
        else:
            p = 4
        df_tt2 = df_tt.replace({0:'F*',1:'s',2:'r',3:'n'})
        model = AgglomerativeClustering(n_clusters=p)
        # model = AgglomerativeClustering(distance_threshold=0,n_clusters=None)
        model = model.fit(df_tt.values)
        # plot_dendrogram(model,truncate_mode='lastp',p=3)
        df_tt['cluster'+str(k)+'('+str(p)+')']=model.labels_
        df_tt2['cluster'+str(k)+'('+str(p)+')']=model.labels_
        df_cl = pd.merge(df_cl,df_tt['cluster'+str(k)+'('+str(p)+')'],how='left',left_index=True,right_index=True)
        df_cl_cr = pd.merge(df_cl_cr,df_tt2,how='left',left_index=True,right_index=True)
        print(k)
        k = k+1
#%%
df_cl_cr.to_excel('cluster/cluster_all_strategy_without_corate.xlsx')
#%%
# TODO 戦略の頻度の（林先生）
df = pd.read_excel('cluster/strategy_total_1.xlsx')
df2 = df.copy()
df2 = df2.iloc[:,:4]
df = df.drop(columns=['day','lv','asg','jdg'])
#%%
p=4
# for j in range(len(df.columns)//15):
#     df_ct = df.iloc[:,j*15:j*15+15]
#     df_ct = df_ct.dropna()
#     for i in range(len(df_ct.columns)//5):
#         df_tt = df_ct.iloc[:,i*5:i*5+5]
        # if any(df_tt.columns.values[0] == col for col in clus3col):
        #     p = 3
        # elif any(df_tt.columns.values[0] == co for co in clus2col):
        #     p = 2
        # else:
        #     p = 4
        # X_distance = gower.gower_matrix(df_t)
model = AgglomerativeClustering(distance_threshold=0,n_clusters=None)
# model = AgglomerativeClustering(distance_threshold=0,n_clusters=None)
model = model.fit(df.values)
plot_dendrogram(model,truncate_mode='lastp',p=p)
plt.title('cluster_strategy_total_with'+str(p)+'cluster')
plt.savefig('cluster_with_strategy_total_with_'+str(p))
plt.show(df.columns.values[0]+str(p)+'cluster')
plt.clf()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = AgglomerativeClustering(n_clusters = p)
model = model.fit(df.values)
#%%
df['cluster('+str(p)+')'] = model.labels_
df3 = pd.merge(df2,df,how='outer',left_index=True,right_index=True)
df3.to_excel('cluster/cluster_strategy_total_1.xlsx')

#%%
df = pd.read_excel('cluster/strategy_total_1.xlsx')
df2 = df.copy()
df2 = df2.iloc[:,:3]
df = df.drop(columns=['day','lv','asg'])
df = df.replace({'f':0,'s':1})

p=4

model = AgglomerativeClustering(distance_threshold=0,n_clusters=None)

model = model.fit(df.values)
plot_dendrogram(model,truncate_mode='lastp',p=p)
plt.title('cluster_strategy_total_with'+str(p)+'cluster_&jdg')
plt.savefig('cluster_with_strategy_total_with_'+str(p)+'&jdg')
plt.show(df.columns.values[0]+str(p)+'cluster')
plt.clf()
#%%
model = AgglomerativeClustering(n_clusters = p)
model = model.fit(df.values)
df['cluster('+str(p)+')'] = model.labels_
df3 = pd.merge(df2,df,how='outer',left_index=True,right_index=True)
df3['jdg'].replace({0:'f',1:'s'},inplace=True)
df3.to_excel('cluster/cluster_strategy_total_1_with_jdg.xlsx')