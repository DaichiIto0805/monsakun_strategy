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

import matplotlib as plt
#%%
files= glob.glob('log_check/monsakun_log_check_r_s??.csv')
print(files)
# モンサクンのチェックと戦略のファイル
for num,file in enumerate(files):
    fnum = re.sub(r"\D","",file)
    fnum = fnum[:2]
    print("fnum=",fnum)

    df = pd.read_csv(file,index_col=0)
    df = df[df['jdg']=='s']
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

    for i in solve_count:
        df1 = df[df['counts']==i]
        df2 = pd.crosstab(df1['strategy'],df1['q'])
        df2.to_csv('strategy_per_solve_count/strategy_per_solve_count_'+str(fnum)+'_'+str(i)+'.csv')