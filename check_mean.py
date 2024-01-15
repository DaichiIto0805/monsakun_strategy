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
df1 = pd.read_csv('monsakun_log_05.csv')
df1 = df1.query('ope1 in ["CHECK"]')
df1 = df1.loc[:,['InputID','check','lv']]
df1_lv1 = df1.query('lv in [1]')
df1_lv2 = df1.query('lv in [2]')
df1_lv3 = df1.query('lv in [3]')
df2 = pd.read_csv('strategy_wide/cluster_strategy_wide_5_meaning.csv',index_col=0)
df2 = df2.sort_index()
df1_lv1 = pd.merge(df1_lv1,df2.loc[:,['InputID','cluster0']],on='InputID')
df1_lv1 = df1_lv1.loc[:,['cluster0','check']].groupby('cluster0').mean()
df1_lv2 = pd.merge(df1_lv2,df2.loc[:,['InputID','cluster1']],on='InputID')
df1_lv2 = df1_lv2.loc[:,['cluster1','check']].groupby('cluster1').mean()
df1_lv3 = pd.merge(df1_lv3,df2.loc[:,['InputID','cluster2']],on='InputID')
df1_lv3 = df1_lv3.loc[:,['cluster2','check']].groupby('cluster2').mean()
# %%
for j in [df1_lv1,df1_lv2,df1_lv3]:
    plt.bar(j.index,j['check'])
    for i in range(len(j.index)):
        plt.text(x=i,y=j['check'][i],s=j['check'][i].round(2),ha='center')
    plt.title(str(j.index.name))
    plt.show()