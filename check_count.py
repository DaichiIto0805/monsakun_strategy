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
df1 = pd.read_csv('monsakun_log_02.csv',index_col=0)
df2 = pd.read_csv('monsakun_log_03.csv',index_col=0)
df3 = pd.read_csv('monsakun_log_04.csv',index_col=0)
df4 = pd.read_csv('monsakun_log_05.csv',index_col=0)

i=2
for df in [df1,df2,df3,df4]:
    df['q']=df['lv'].astype(str).str.cat(df['asg'].astype(str),sep='_')
    df['q']=str(i)+'_'+df['q']
    df.insert(4,'story_num',i)
    i+=1


df5 = pd.concat([df1,df2,df3,df4])

df0 = pd.read_csv('monsakun_log_per_solve_counts/monsakun_log_per_solve_counts_all.csv')
df0 = df0[['InputID','lv','asg','strategy']]

dfm = pd.merge(df0,df5,on=['InputID','lv','asg'],how='left')

df6 = dfm[dfm['ope1']=='CHECK']
df6 = df6[df6['jdg']=='s']
df7 = pd.pivot_table(df6,index='InputID',columns='q',values='check')
df7.to_csv('check/check_cross.csv')

df8 = pd.merge()
# %%
