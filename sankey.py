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
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%df1 = pd.read_csv('monsakun_log_05.csv')
df1 = pd.read_csv('monsakun_log_05.csv')
fname = 'strategy_wide/cluster_strategy_wide_5_meaning.csv'
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
df4 = df4[df4['cluster'].isin([0,2,6])]
for i in range(1,4):
    df5 = df4[df4['lv']==i]
    df6 = pd.pivot_table(df5,index='lv_asg',columns='cluster',values='stp')
    print(df4.columns)
    df6.plot(marker='.')
    plt.ylabel('stp')
    plt.ylim(2,16)
    plt.title('file' + str(fnum)+'_lv'+str(i))
    # plt.savefig('strategy_wide/strategy_stp_file'+str(fnum)+'_lv'+str(i)+'cl016.png')
    # plt.savefig('strategy_wide/strategy_stp_file'+str(fnum)+'_lv'+str(i)+'cl12.png')
    # plt.savefig('strategy_wide/strategy_stp_file'+str(fnum)+'_lv'+str(i)+'cl123.png')
    plt.savefig('strategy_wide/strategy_stp_file'+str(fnum)+'_lv'+str(i)+'cl026.png')
    plt.show()
# %%
from io import StringIO
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
#%%
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

fig.write_image('strategy_wide/sankey_day_meaning'+str(fnum)+'.png')
# %%
