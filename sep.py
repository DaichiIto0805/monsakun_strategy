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
#%%
fname = "./strategy_wide/cluster_strategy_wide_4.csv"
fnum = re.sub(r'\D', '', fname)
df = pd.read_csv(fname,index_col=0)
df = df.sort_index()
# df_concat = pd.concat([df, df["1_1"].str.split("_", expand=True)], axis=1)
for i in range(1,4):
    for j in range(1,6):
        df[[str(i)+'_'+str(j)+'_F',str(i)+'_'+str(j)+'_S',str(i)+'_'+str(j)+'_R']] = df[str(i)+'_'+str(j)].str.split('_',expand=True)
        df.drop([str(i)+'_'+str(j)],axis=1,inplace=True)

# print(df.columns.values)
df=df.reindex(columns=['InputID','day','1_1_F','1_1_S','1_1_R'
,'1_2_F','1_2_S','1_2_R','1_3_F','1_3_S','1_3_R','1_4_F','1_4_S','1_4_R'
,'1_5_F','1_5_S','1_5_R','cluster0','2_1_F','2_1_S','2_1_R','2_2_F','2_2_S','2_2_R'
,'2_3_F','2_3_S','2_3_R','2_4_F','2_4_S','2_4_R','2_5_F','2_5_S','2_5_R','cluster1'
,'3_1_F','3_1_S','3_1_R','3_2_F','3_2_S','3_2_R','3_3_F','3_3_S','3_3_R'
,'3_4_F','3_4_S','3_4_R','3_5_F','3_5_S','3_5_R','cluster2'])
#%%
df.to_csv('strategy_wide/strategy_wide_'+str(fnum)+'_sep.csv')
# %%
for i in range(3):
    df1 = df.loc[:,str(i+1)+'_1_F':'cluster'+str(i)]
    df1 = df1.replace({'x':0,'F':1,'R':1,'S':1})
    # df2 = df1.groupby('cluster0').apply(np.sum)
    df2 = df1.groupby('cluster'+str(i)).apply(lambda x:x.sum())
    grouped_sum = df1.groupby('cluster'+str(i)).apply(lambda x: x.iloc[:, :-1].sum())
    grouped_size = df1.groupby('cluster'+str(i)).size()

    df3 = grouped_sum.copy()
    for col in grouped_sum[:-1]:
        df3[col] = grouped_sum[col]/grouped_size
    df3.to_csv('strategy_wide/cluster_strategy_ratio'+str(fnum)+'_lv_'+str(i+1)+'_.csv')


# %%
l = glob.iglob('strategy_wide/cluster_strategy_ratio?_lv_?_.csv')
for i in l:
    m = re.findall(r'\d+',i)[0]
    m2 = re.findall(r'\d+',i)[1]
    df0 = pd.read_csv(i,index_col=0)
    df100 = df0 * 100
    df100.to_csv('strategy_wide/cluster_strategy_ratio_'+str(m)+'_lv_'+str(m2)+'_100.csv')