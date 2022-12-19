# %%
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

sns.set(font='IPAexGothic')

# %%
import os
dir = ['whole_level_graphs','strategy_lv']
for i in range(len(dir)):
    if not os.path.exists(dir[i]):
        os.mkdir(dir[i])

# %%
from matplotlib.ticker import MaxNLocator
files = glob.glob('log_check/monsakun_log_check_r_s??.csv')
for file1 in files:
    fnum1 = re.sub(r"\D","",file1)
    df1 = pd.read_csv('InputID_q_drop_'+str(fnum1)+'.csv',index_col=0)
    df2 = pd.read_csv(file1,index_col=0)
    df2 = df2[df2['check']==1]
    df3 = pd.merge(df2,df1['InputID'],on=['InputID'],how='inner')
    df4 = pd.crosstab([df3['relation_st'],df3['story_st'],df3['formula_st']],df3['lv'])

    fig, ax = plt.subplots(figsize=(10,8))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))



    #積み上げ棒グラフ描画
    st_list = [(0,0,0), (0,0,1),(0,1,0), (0,1,1), (1,0,0), (1,0,1)]
    colorlist = mcolors.TABLEAU_COLORS.keys()

    dict_colorlist = dict(zip(st_list,colorlist))

    for i in range(len(df4)):
        ax.bar(df4.columns, df4.iloc[i], bottom=df4.iloc[:i].sum(),color=dict_colorlist[df4.index[i]])
        for j in range(len(df4.columns)):
            plt.text(x=j+1, 
                        y=df4.iloc[:i, j].sum() + (df4.iloc[i, j] / 2), 
                        s=df4.iloc[i, j], 
                        ha='center', 
                        va='bottom'
                    )
    ax.set(xlabel='q', ylabel='strategy')
    ax.legend(df4.index)
    plt.title('first_check_whole_lv_'+str(fnum1))
    plt.savefig('whole_level_graphs/r_s_t_first_check_whole_lv'+str(fnum1)+'.png')
    plt.show()
# %%
files = glob.glob('log_check/monsakun_log_check_r_s??.csv')
for file1 in files:
    fnum1 = re.sub(r"\D","",file1)
    df1 = pd.read_csv('InputID_q_drop_'+str(fnum1)+'.csv',index_col=0)
    df2 = pd.read_csv(file1,index_col=0)
    df2 = df2[df2['check']==1]
    df3 = pd.merge(df2,df1['InputID'],on=['InputID'],how='inner')

    #ゴリ押しで整形（戦略のワンホットベクトルを直す）
    query_str = "story_st == 1 and formula_st == 1"
    df3_subset = df3.query(query_str)
    df3.loc[df3_subset.index,"story_st"]=0
    query_str = "relation_st == 1 and formula_st == 1"
    df3_subset = df3.query(query_str)
    df3.loc[df3_subset.index,"relation_st"]=0
    df3.relation_st[df3.relation_st==1]='r'
    df3.relation_st[df3.relation_st==0]=''
    df3.story_st[df3.story_st==1]='s'
    df3.story_st[df3.story_st==0]=''
    df3.formula_st[df3.formula_st==1]='F*'
    df3.formula_st[df3.formula_st==0]=''
    df3['strategy']=df3['relation_st']+df3['story_st']+df3['formula_st']
    df3.strategy[df3.strategy=='']='n'
    df3.to_csv('log_check/log_strategy_each_value_'+str(fnum1)+'.csv')

    df4 = pd.crosstab(df3['strategy'],df3['lv'])

    ig, ax = plt.subplots(figsize=(10,8))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))



    #積み上げ棒グラフ描画
    storategy_list = ['F*','n','r','s']
    colorlist = mcolors.TABLEAU_COLORS.keys()
    dict_colorlist = dict(zip(storategy_list,colorlist))

    for i in range(len(df4)):
        ax.bar(df4.columns, df4.iloc[i], bottom=df4.iloc[:i].sum(),color=dict_colorlist[df4.index[i]])
        for j in range(len(df4.columns)):
            plt.text(x=j+1, 
                        y=df4.iloc[:i, j].sum() + (df4.iloc[i, j] / 2), 
                        s=df4.iloc[i, j], 
                        ha='center', 
                        va='bottom'
                    )
    ax.set(xlabel='q', ylabel='strategy')
    ax.legend(df4.index)
    plt.title('first_check_strategy_lv_'+str(fnum1))
    plt.savefig('whole_level_graphs/r_s_t_first_check_whole_lv_2_'+str(fnum1)+'.png')
    plt.show()
#%%[markdown]
# χ二乗検定
# #%%
# from scipy import stats
# def print_chi(data):

#     chi2,p,dof,exp = stats.chi2_contingency(data,correction='Bonferroni')
#     print('期待度数',"\n",exp)
#     print('自由度',"\n",dof)
#     print('カイ二乗値',"\n",chi2)
#     print('p値',"\n",p)

# df_chi=df4
# print_chi(df_chi)
#%%
import pandas as pd
import numpy as np
from scipy.stats import norm, chi2, chi2_contingency
import statsmodels.stats.multitest as multi



def residual_analysis(table: pd.DataFrame, p_value: int=0.05):
    """
    クロス集計結果に対して残差分析を実施し、指定したp値以下の組み合わせを取得するメソッド。
    
    Parameters
    -------
    table : pd.DataFrame
        クロス集計結果。インデックス、カラム名を指定すること。
    p_value : int
        p値。
    
    Returns
    -------
    pair list : list
        インデックス、カラム名の組み合わせtupleのlist
    
    """
    
    # numpy.arrayに変換
    np_data = np.array(data)
    
    # カイ二乗検定
    chi_sqared, chi_p_value, df, exp = chi2_contingency(np_data,correction=False)
    multi.multipletests(chi_p_value, alpha=0.05, method="holm") 

    if chi_p_value < p_value:
        print(f'カイ二乗検定：有意水準{p_value}で有意差あり。({chi_p_value})')
    else:
        print(f'カイ二乗検定：有意水準{p_value}で有意差なし。({chi_p_value})')
    
    # インデックスとカラム名
    index = data.index
    column = data.columns
    
    # 行数と列数を取得
    row_num, col_num = np_data.shape
    # 合計
    total = np_data.sum()
    # 行と列ごとの合計
    total_by_row = [np_data[i, :].sum() for i in range(row_num)]
    total_by_col = [np_data[:, i].sum() for i in range(col_num)]
    
    # 期待値
    exp = np.array(exp)
    
    pairs = list()
    # 期待値と残差分散を算出
    for i in range(row_num):
        for j in range(col_num):
            # 残差分散
            res_var = (1 - total_by_row[i]/total)*(1 - total_by_col[j]/total)
            # 調整済み標準化残差
            std_res = (np_data[i, j] - exp[i, j])/np.sqrt(exp[i, j] * res_var)
            # 両側検定
            p = norm.sf(x=abs(std_res), loc=0, scale=1)*2
            # p値を下回るペア
            if p <= p_value:
                pairs.append((index[i], column[j]))
    return pairs

files = glob.glob('log_check/log_strategy_each_value_??.csv')
for file1 in files:
    df1 = pd.read_csv(file1)
    df2 = pd.crosstab(df1['strategy'],df1['lv'])
    print(file1)
    data = df2
    print(residual_analysis(table=data, p_value=0.05))