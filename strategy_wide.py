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
fname = 'strategy_wide/strategy_wide_4.csv'
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
p=12

for j in range(len(df.columns)//15):
    df_ct = df.iloc[:,j*15:j*15+15]
    df_ct = df_ct.dropna()
    for i in range(len(df_ct.columns)//5):
        df_ctt = df_ct.iloc[:,i*5:i*5+5]
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
        print(k,p)
        k = k+1
fnum = re.sub(r"\D","",fname)
print(data_dict)
df_cl_cr.to_csv('strategy_wide/cluster_strategy_wide_'+fnum+'_'+str(p)+'clusters.csv')
df_cl_cr.to_excel('strategy_wide/cluster_strategy_wide_'+fnum+'_'+str(p)+'clusters.xlsx')

#%%%%%%%%%%%%
df1 = pd.read_csv('monsakun_log_04.csv')
fname = 'strategy_wide/cluster_strategy_wide_4_meaning.csv'
df2 = pd.read_csv(fname)
df1 = df1[df1['ope1']=='CHECK']
df1['No']=range(0,len(df1.index))
df2 = df2.filter(regex ='^(cluster|InputID)',axis=1)
fnum = re.sub(r'\D', '', fname)
df4 = df1.iloc[0:0]
for i in range(3):
    df1c = df1.copy()
    df1c = df1c[df1c['lv']==i+1]
    df2c = df2.copy()
    df2c = df2c.loc[:,['InputID','cluster'+str(i)]]

    df3 = pd.merge(df1c,df2c,how='left',on='InputID')
    df3 = df3.rename(columns={'cluster'+str(i):'cluster'})
    df4 = df4.append(df3)

df4 = df4.sort_values('No')
df4 = df4.filter(regex='^(cluster|lv|InputID|stp|asg)',axis=1)
df4 = df4.dropna(subset=['cluster'])
df4['cluster']=df4['cluster'].astype(pd.Int64Dtype(),errors='ignore')
# df4['cluster'] = 'cl_' + df4['cluster'].astype(str)
df4['lv_asg'] = df4['lv'].astype(str).str.cat(df4['asg'].astype(str),sep='_')

for i in range(1,4):
    df5 = df4[df4['lv']==i]
    df6 = pd.pivot_table(df5,index='lv_asg',columns='cluster',values='stp')
    print(df4.columns)
    df6.plot(marker='.')
    plt.ylabel('stp')
    plt.title('file' + str(fnum)+'_lv'+str(i))
    plt.savefig('strategy_wide/strategy_stp_file_meaning'+str(fnum)+'_lv'+str(i)+'.png')
    plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from io import StringIO
import plotly.graph_objects as go
#%%%
df4_1 = df4[['InputID','lv','cluster']].drop_duplicates()
df7 = df4.pivot_table(index='InputID',columns='lv',values='cluster')
df7 = df7.sort_values(by=[1,2,3])
df7[1]=df7[1].astype(pd.Int64Dtype(),errors='ignore')
df7[2]=df7[2].astype(pd.Int64Dtype(),errors='ignore')
df7[3]=df7[3].astype(pd.Int64Dtype(),errors='ignore')
df7[1]='1_'+df7[1].astype(str)
df7[2]='2_'+df7[2].astype(str)
df7[3]='3_'+df7[3].astype(str)
df7 = df7.stack()
# df7.reset_index(inplace=True)
df8 = df7.reset_index()
df8 = df8[['InputID',0]]
df8['to']=df8[0].shift(-1)[df8['InputID']==df8['InputID'].shift(-1)].dropna()
df8 = df8.rename(columns={0:'from'}).drop(columns=['InputID'])
df8 = df8.dropna()
df8 = df8.groupby(['from','to']).size().reset_index(name='value')
# df7['to'] = df7[0].shift(-1)[df7['ID'] == df7['ID'].shift(-1)].dropna()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

fig.write_image('strategy_wide/sankey_day'+str(fnum)+'.png')
# %%
