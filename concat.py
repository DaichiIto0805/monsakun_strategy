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

sns.set(font='IPAexGothic')
#%%
files= glob.glob('log_check/monsakun_log_check_r_s??.csv')
print(files)
# モンサクンのチェックと戦略のファイル
for num,file in enumerate(files):
    fnum = re.sub(r"\D","",file)
    fnum = fnum[:2]
    print("fnum=",fnum)

    df = pd.read_csv(file,index_col=0)
    # 重複削除
    df = df.drop_duplicates(subset=['InputID','q'])
    # display(df)
    vc = df['InputID'].value_counts()
    vc = vc.rename('counts')
    df = pd.merge(df,vc,left_on='InputID',right_index=True)
    # vc = pd.unique(df['counts'])
    solve_count = pd.unique(df['counts'])

    query_str = "story_st == 1 and formula_st == 1"
    df_subset = df.query(query_str)
    df.loc[df_subset.index,"story_st"]=0
    query_str = "relation_st == 1 and formula_st == 1"
    df_subset = df.query(query_str)
    df.loc[df_subset.index,"relation_st"]=0
    df.relation_st[df.relation_st==1]='r'
    df.relation_st[df.relation_st==0]=''
    df.story_st[df.story_st==1]='s'
    df.story_st[df.story_st==0]=''
    df.formula_st[df.formula_st==1]='F*'
    df.formula_st[df.formula_st==0]=''
    df['strategy']=df['relation_st']+df['story_st']+df['formula_st']
    df.strategy[df.strategy=='']='n'

    df.to_csv('monsakun_log_per_solve_counts_fs/monsakun_log_per_solve_counts_fs_'+str(fnum)+'.csv')
#%%
files= glob.glob('log_check/monsakun_log_check_r_s??.csv')
print(files)
# モンサクンのチェックと戦略のファイル
df1 = pd.read_csv('monsakun_log_per_solve_counts_fs/monsakun_log_per_solve_counts_fs_02.csv',index_col=0)
df2 = pd.read_csv('monsakun_log_per_solve_counts_fs/monsakun_log_per_solve_counts_fs_03.csv',index_col=0)
df3 = pd.read_csv('monsakun_log_per_solve_counts_fs/monsakun_log_per_solve_counts_fs_04.csv',index_col=0)
df4 = pd.read_csv('monsakun_log_per_solve_counts_fs/monsakun_log_per_solve_counts_fs_05.csv',index_col=0)

i=2
for df in [df1,df2,df3,df4]:
    df['q']=str(i)+'_'+df['q']
    df.insert(4,'story_num',i)
    i+=1

df5 = pd.concat([df1,df2,df3,df4])
df5 = df5.drop('counts',axis=1)

vc = df5['InputID'].value_counts()
vc = vc.rename('counts')
df5 = pd.merge(df5,vc,left_on='InputID',right_index=True)
# vc = pd.unique(df['counts'])
solve_count = pd.unique(df5['counts'])

df5.to_csv('monsakun_log_per_solve_counts/monsakun_log_per_solve_counts_all.csv')
# df5 = df5[df5['counts']==60]
df5['strategy_jdg'] = df5['strategy'].str.cat(df5['jdg'],sep='_')
# df5 = df5.set_index(['InputID','counts'])
# a = df5['InputID'].unique()
df6 = df5.loc[:,['InputID','counts']]
df6 = df6.drop_duplicates()
df6 = df6.set_index('InputID')
df = df5.pivot(index='InputID',columns='q',values='strategy_jdg')
df = pd.merge(df6,df,how='left',left_index=True,right_index=True)
df = df.sort_values('counts',ascending=False)
df5.to_csv('strategy_all_cross/all_mosnakun_all_check1.csv')
df.to_csv('strategy_all_cross/strategy_all_cross.csv')
# %%
