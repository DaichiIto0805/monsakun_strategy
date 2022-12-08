

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

import matplotlib as plt
# %%
import os
dir = ['graphs','first_check','log','log_check','first_check','exact_data']
for i in range(len(dir)):
    if not os.path.exists(dir[i]):
        os.mkdir(dir[i])

# %%
file = 'check_monsakun_log_02.csv'
df = pd.read_csv(file,index_col=0)
df=df.dropna(0)
df2 = df.mean()
# display(df2)
df2.to_csv('check_monsakun_mean_02.csv')

# %%
#　物語戦略
files1= glob.glob('monsakun_log_??.csv')
files2= glob.glob('problems_shobi2017-?.csv')
for num1,file1 in enumerate(files1):
  fnum1 = re.sub(r"\D","",file1)
  print("fnum1=",fnum1)
  fnum2 = fnum1[-1]
  
  
  file2 = 'problems_shobi2017-'+str(fnum2)+'.csv'
  print(file1,file2)
  df1 = pd.read_csv(file1)
  df2 = pd.read_csv(file2)

  df3 = pd.merge(df1,df2,on=['lv','asg','card'],how='left')
  df3['story_st']=0
  df3['relation_st']=0
  df3['formula_st']=0
  # df.loc[(df['stp']==1)&(df['card']==3),'relation_st']=1
  df3.loc[(df3['stp']==1)&(df3['type0']=='r1'),'relation_st']=1
  df3.loc[(df3['stp']==1)&(df3['type0']=='r0'),'relation_st']=1

  for index,row in df3.iterrows():
    if row['stp']==1:
      relation_st = row['relation_st']
    elif row['ope1']!='PROB_BEGIN':
      df3.loc[index,'relation_st']=relation_st

  set_ct = 0
  for index,row in df3.iterrows():
    # print(row['session'],session,row['ope1'],row['check'],row['type2'])
    if row['ope1']=='PROB_BEGIN':
      session = row['session']
      set_ct = 0
      story1 = 0
      story2 = 0
      formula = 0
    if row['session'] == session and row['ope1']=='SET' and row['check']==0:
      # print(index,set_ct,row['type2'],row['slot'])
      set_ct += 1
      # print(set_ct)
      if set_ct == 1 and row['type2'] =='existence' and row['slot']==1:
        story1 += 1
        story2 += 1
      elif set_ct == 2 and row['type2'] in ['increase','decrease'] and row['slot'] == 2:
        story1 += 1
      elif set_ct == 2 and row['type2'] =='existence' and row['slot'] == 2:
        story2 += 1
      elif set_ct == 3 and (row['type2'] in ['combination','comp_more']) and row['slot']==3:
        story2 +=1
      elif set_ct == 3 and (row['type2'] =='existence') and row['slot']==3:
        story1 +=1

      if set_ct == row['formula_order']==row['slot']:
        formula += 1
    if row['session'] == session and row['ope1']=='CHECK':
      if story2 ==3 or story1 == 3:
        # for index,row in df3.iterrows():
        #   if session == row['session']:
        #     df3.loc[index,'story_st']=1
        df3.loc[index,'story_st']=1

      if formula == 3:
        df3.loc[index,'formula_st']= '1'
 
  df3.to_csv('log/monsakun_log_'+str(fnum1)+'_with_type.csv')
  df3['q']=df3['lv'].astype(str).str.cat(df3['asg'].astype(str),sep='_')
  df4 = df3.loc[df3['ope1']=='CHECK']
  df4.to_csv('log_check/monsakun_log_check_r_s'+str(fnum1)+'.csv')
  df4 = df4[df4['check']==1]
  pd.crosstab([df4['relation_st'],df4['story_st']],[df4['q'],df4['jdg']]).to_csv('first_check/r_s_first_check_'+str(fnum1)+'.csv')
  pd.crosstab([df4['relation_st'],df4['story_st']],df4['q'],margins=True).to_csv('first_check/r_s_first_check_no_jdg_'+str(fnum1)+'.csv')
  pd.crosstab([df4['relation_st'],df4['story_st'],df4['formula_st']],df4['q'],margins=True).to_csv('first_check/r_s_f_first_check_no_jdg_'+str(fnum1)+'.csv')
  pd.crosstab([df4['InputID']],df4['q'],margins=True).to_csv('InputID_q_'+str(fnum1)+'.csv')

# %%
def remove_any_zero_row(df):
    """一つでも0の行を削除"""
    df = df.copy()
    for row in df.index:
        if (df.loc[row] == 0).any():
            df.drop(row, axis=0, inplace=True)
    return df

# %%
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pprint

# %%
files1= glob.glob('log_check/monsakun_log_check_r_s??.csv')
for num1,file1 in enumerate(files1):
  fnum1 = re.sub(r"\D","",file1)
  print("fnum1=",fnum1)
  fnum2 = fnum1[-1]
  
  
  file2 = 'InputID_q_'+str(fnum1)+'.csv'
  print(file1,file2)
  df1 = pd.read_csv(file1,index_col=0)
  df2 = pd.read_csv(file2)
  df2 = remove_any_zero_row(df2)
  df2 = df2[:-1]
  df2 = df2.drop(columns=['All'])
  df2.to_csv('InputID_q_drop_'+str(fnum1)+'.csv')
  df1 = df1[df1['check']==1]
  df3 = pd.merge(df1,df2['InputID'],on=['InputID'],how='inner')
  df3.to_csv('log_check/log_check_InputID_'+str(fnum1)+'.csv')
  df4 = pd.crosstab([df3['relation_st'],df3['story_st'],df3['formula_st']],df3['q'])
  df4.to_csv('first_check/r_s_f_first_check_do_all_no_sum_'+str(fnum1)+'.csv')
  pd.crosstab([df3['relation_st'],df3['story_st']],df3['q'],margins=True).to_csv('first_check/r_s_f_first_check_do_all_'+str(fnum1)+'.csv')
  fig, ax = plt.subplots(figsize=(10, 8))


  #積み上げ棒グラフ描画
  st_list = [(0,0,0), (0,0,1),(0,1,0), (0,1,1), (1,0,0), (1,0,1)]
  colorlist = mcolors.TABLEAU_COLORS.keys()

  dict_colorlist = dict(zip(st_list,colorlist))

  for i in range(len(df4)):
      ax.bar(df4.columns, df4.iloc[i], bottom=df4.iloc[:i].sum(),color=dict_colorlist[df4.index[i]])
      for j in range(len(df4.columns)):
        plt.text(x=j, 
                 y=df4.iloc[:i, j].sum() + (df4.iloc[i, j] / 2), 
                 s=df4.iloc[i, j], 
                 ha='center', 
                 va='bottom'
                )
  ax.set(xlabel='q', ylabel='strategy')
  ax.legend(df4.index)
  plt.title('r_s_t_first_check_do_all_'+str(fnum1))
  plt.savefig('graphs/r_s_t_first_check_do_all_'+str(fnum1)+'.png')
  plt.show()

# %% [markdown]
# 直接確率の算出

# %%
files = glob.glob('first_check/r_s_f_first_check_do_all_no_sum_??.csv')
e_data = pd.DataFrame(columns = ['郡１','群２','観測値１','観測値２','群1_観測値1','群１観測値２','群２観測値１','群２観測値２','p値'])
for file in files:
  fnum = re.sub(r"\D","",file)
  print(fnum)
  df = pd.read_csv(file,index_col=['relation_st','story_st','formula_st'])
  print(print(len(df.index)))
  df2 = df.index
  df_index = []
  for element in df2:
    element = '_'.join(map(str,element))
    df_index.append(element)
    print(element)
  print(df_index)
  for i in range(len(df)-1):
    for j in range(len(df.columns)-1):
      data = [[df.iloc[i,j],df.iloc[i,j+1]],[df.iloc[i+1,j],df.iloc[i+1,j+1]]]
      # print(data)
      # print(df.index.values[i],df.columns.values[j],stats.fisher_exact(data)[1])
      tmp_se = pd.Series([df_index[i],df_index[i+1],str(df.columns[j]),str(df.columns[j+1]),df.iloc[i,j],df.iloc[i,j+1],df.iloc[i+1,j],df.iloc[i+1,j+1],stats.fisher_exact(data)[1]],index=e_data.columns)
      tmp_seT = tmp_se.T
      e_data = e_data.append(tmp_se,ignore_index=True)
      # e_data = pd.concat([e_data,tmp_se],axis=1)
      # print(list(itertools.chain.from_iterable(data)))
  display(e_data)
  e_data.to_csv('exact_data/exact_data_'+str(fnum)+'.csv')
  # print(df.index[1])

# %%
# pd.crosstab([df3['relation_st'],df3['story_st']],[df3['q'],df3['jdg']]).to_csv('r_s_first_check_'+'02')+'.csv')
#新しい列の作成（レベル＿問題番号）
df3['q']=df3['lv'].astype(str).str.cat(df3['asg'].astype(str),sep='_')
df3.loc[df['ope1']=='CHECK'].to_csv('monsakun_log_check_r_s'+str(fnum)+'.csv')
pd.crosstab([df3['relation_st'],df3['story_st']],[df3['q'],df3['jdg']]).to_csv('r_s_first_check_'+'02'+'.csv')


# %%
files= glob.glob('monsakun_log_??_check.csv')
for num,file in enumerate(files):
  fnum = re.sub(r"\D","",file)
  fnum = fnum[:2]
  print("fnum=",fnum)

  df = pd.read_csv(file)
  
  display(df[['lv','check']].groupby('lv').mean())
  # print(df['lv'])

# %%
files= glob.glob('monsakun_log_??.csv')
for num,file in enumerate(files):
  fnum = re.sub(r"\D","",file)
  fnum = fnum[:2]
  print("fnum=",fnum)

  df = pd.read_csv(file)
  df['relation_st']=0
  df['story_st']=0
  # display(df)

#関係文の定義
  # if fnum == '02' or fnum == '05':
  #   df.loc[(df['stp']==1)&(df['card']==3),'relation_st']='card_3'
  #   df.loc[(df['stp']==1)&(df['card']==5),'relation_st']='card_5'
  # else:
  #   df.loc[(df['stp']==1)&(df['card']==2),'relation_st']='card_2'
  #   df.loc[(df['stp']==1)&(df['card']==5),'relation_st']='card_5'
  if fnum == '02' or fnum == '05':
    df.loc[(df['stp']==1)&(df['card']==3),'relation_st']=1
    df.loc[(df['stp']==1)&(df['card']==5),'relation_st']=1
  else:
    df.loc[(df['stp']==1)&(df['card']==2),'relation_st']=1
    df.loc[(df['stp']==1)&(df['card']==5),'relation_st']=1

  for index, row in df.iterrows():
    
    if row['ope1']=='CHECK':
      check = 1
    else:
      check = 0
    if row['ope1']=='PROB_BEGIN':
      begin_session = row['session']
      set1 = 0
      set2 = 0
      set3 = 0
      cond1 = 0
      cond2 = 0
      set_ct = 0
    # print('set_ct',set_ct)
    elif row['ope1']=='SET':
      set_ct += 1
      if fnum == '02' or fnum =='05':
        if set_ct == 1 and row['slot1'] not in [0,3,5] and row['slot2'] == 0\
          and row['slot3'] == 0:
          cond1 += 1
          cond2 += 1
        elif set_ct == 2 and row['slot1'] not in [0,3,5] and row['slot2'] not in [0,3,5]\
              and row['slot3'] == 0:
          cond1 += 1
        elif set_ct == 2 and row['slot1'] not in [0,3,5] and row['slot2'] == 5 and \
              row['slot3'] == 0:
          cond2 += 1
        elif set_ct == 3 and row['slot1'] not in [0,3,5] and row['slot2'] ==5 and\
              row['slot3'] not in [0,3,5]:
          cond2 += 1
        elif set_ct ==3 and row['slot1'] not in [0,3,5] and row['slot2'] not in [0,3,5]\
              and row['slot3'] ==3:
          cond1 +=1
        # elif row['slot1'] not in [0,3,5]:
        #   if row['slot2']==5:
        #     if row['slot3'] not in [0,3,5]:
        #       df.loc[index,'story_st']=1
      else:
        if set_ct == 1 and row['slot1'] not in [0,2,5] and row['slot2'] == 0\
          and row['slot3'] == 0:
          cond1 += 1
          cond2 += 1
        elif set_ct == 2 and row['slot1'] not in [0,2,5] and row['slot2'] not in [0,2,5]\
              and row['slot3'] == 0:
          cond1 += 1
        elif set_ct == 2 and row['slot1'] not in [0,2,5] and row['slot2'] == 2 and \
              row['slot3'] == 0:
          cond2 += 1
        elif set_ct == 3 and row['slot1'] not in [0,2,5] and row['slot2'] ==2 and\
              row['slot3'] not in [0,2,5]:
          cond2 += 1
        elif set_ct ==3 and row['slot1'] not in [0,2,5] and row['slot2'] not in [0,2,5]\
              and row['slot3'] ==5:
          cond2 += 1
    elif row['ope1']=='CHECK':
      if cond1 == 3 or cond2 ==3:
        df.loc[index,'story_st']=1

    if row['relation_st']!=0:
      relation_st = row['relation_st']
      session = row['session']
      # print(relation_st)  
      # print(row['session'],session)
    if (row['session']==session):
      if(row['stp']!=1):
        df.loc[index,'relation_st']=relation_st
  df.to_csv('monsakun_log_relation_story_'+str(fnum)+'.csv')
  df.loc[df['ope1']=='CHECK'].to_csv('monsakun_log_check_relation_story_'+str(fnum)+'.csv')

# %%
# 関係文戦略と物語戦略の比較
files = glob.glob('monsakun_log_check_relation_story_??.csv')
for index,file in enumerate(files):
  fnum = re.sub(r"\D","",file)
  fnum = fnum[:2]
  print("fnum=",fnum)

  df = pd.read_csv(file,index_col=0)
  # df['relation_st'] = 0

  # #関係文の定義
  # if fnum == '02' or fnum == '05':
  #   df.loc[(df['stp']==1)&(df['card']==3),'relation_st']='card_3'
  #   df.loc[(df['stp']==1)&(df['card']==5),'relation_st']='card_5'
  # else:
  #   df.loc[(df['stp']==1)&(df['card']==2),'relation_st']='card_2'
  #   df.loc[(df['stp']==1)&(df['card']==5),'relation_st']='card_5'

  #新しい列の作成（レベル＿問題番号）
  df['q']=df['lv'].astype(str).str.cat(df['asg'].astype(str),sep='_')

  #チェック数１での結果(関係文と物語)
  df2 = df[df['check']==1]
  df2.to_csv('check_1_'+str(fnum)+'.csv')
  pd.crosstab(df2['relation_st'],[df2['q'],df2['jdg']]).to_csv('relation_first_check_'+str(fnum)+'.csv')
  pd.crosstab([df2['relation_st'],df2['story_st']],[df2['q'],df2['jdg']]).to_csv('relation_story_first_check_'+str(fnum)+'.csv')
  pd.pivot_table(df2,index='InputID',columns=['q'],values=['relation_st','story_st']).to_csv('InputID_q_relation_story'+str(fnum)+'.csv')
  pd.crosstab(df2['InputID'],[df2['q']]).to_csv('InputID_q_relation_story'+str(fnum)+'.csv')
  pd.crosstab(df2['relation_st'],df2['q'],margins=True).to_csv('relation_st_per_q_'+str(fnum)+'.csv')

  #戦略ごとのチェック数
  df2 = df[df['jdg']=='s']
  # if index == 0:
  df2.to_csv('relation_st'+str(fnum)+'.csv')


  pd.pivot_table(df2,index='relation_st',columns=['q'],values='check',aggfunc=[np.mean,len]).to_csv('strategy_check_'+str(fnum)+'.csv')
  pd.pivot_table(df2,index='relation_st',columns=['q'],values='stp',aggfunc=[np.mean,len]).to_csv('strategy_step_'+str(fnum)+'.csv')

# %%
files = glob.glob('monsakun_log_check_relation_story_02.csv')

for index,file in enumerate(files):
  fnum = re.sub(r"\D","",file)
  fnum = fnum[:2]
  print("fnum=",fnum)

  df = pd.read_csv(file,index_col=0)
  df = df[df['check']==1]
  df.to_csv('02.csv')
  df = df.replace({'relation_st':{1:'relation_st'},'story_st':{1:'story_st'}})
  print(df)
  df['strategy']=df['relation_st'].astype(str).str.cat(df['story_st'].astype(str))
  df['q'] = df['lv'].astype(str).str.cat(df['asg'].astype(str),sep='_')
  df = df.replace({'strategy':{'0*':''}},regex=True)
  df = df.replace({'strategy':{'':'0'}},regex=True)
  # print(df)
  pd.crosstab(df['InputID'],[df['q'],df['strategy'],df['jdg']]).to_csv('InputID_relation_story_'+str(fnum)+'.csv')
  # display(df)

# %%

files = glob.glob('monsakun_log_check_relation_st_??.csv')
for index,file in enumerate(files):
  fnum = re.sub(r"\D","",file)
  fnum = fnum[:2]
  print("fnum=",fnum)

  df = pd.read_csv(file,index_col=0)
  # df['relation_st'] = 0

  # #関係文の定義
  # if fnum == '02' or fnum == '05':
  #   df.loc[(df['stp']==1)&(df['card']==3),'relation_st']='card_3'
  #   df.loc[(df['stp']==1)&(df['card']==5),'relation_st']='card_5'
  # else:
  #   df.loc[(df['stp']==1)&(df['card']==2),'relation_st']='card_2'
  #   df.loc[(df['stp']==1)&(df['card']==5),'relation_st']='card_5'

  #新しい列の作成（レベル＿問題番号）
  df['q']=df['lv'].astype(str).str.cat(df['asg'].astype(str),sep='_')

  #チェック数１での結果
  df2 = df[df['check']==1]
  df2.to_csv('check_1_'+str(fnum)+'.csv')
  pd.crosstab(df2['relation_st'],[df2['q'],df2['jdg']]).to_csv('relation_first_check_'+str(fnum)+'.csv')
  pd.crosstab([df2['relation_st'],df2['story_st']],[df2['q'],df2['jdg']]).to_csv('relation_story_first_check_'+str(fnum)+'.csv')
  pd.pivot_table(df2,index='InputID',columns=['q'],values=['relation_st']).to_csv('InputID_q_'+str(fnum)+'.csv')
  pd.crosstab(df2['relation_st'],df2['q'],margins=True).to_csv('relation_st_per_q_'+str(fnum)+'.csv')

  #戦略ごとのチェック数
  df2 = df[df['jdg']=='s']
  # if index == 0:
  df2.to_csv('relation_st'+str(fnum)+'.csv')


  pd.pivot_table(df2,index='relation_st',columns=['q'],values='check',aggfunc=[np.mean,len]).to_csv('strategy_check_'+str(fnum)+'.csv')
  pd.pivot_table(df2,index='relation_st',columns=['q'],values='stp',aggfunc=[np.mean,len]).to_csv('strategy_step_'+str(fnum)+'.csv')

# %%
files = glob.glob('relation_st??.csv')
usr = {}
for index,file in enumerate(files):
  fnum = re.sub(r"\D","",file)
  fnum = fnum[:2]
  print("fnum=",fnum)

  df = pd.read_csv(file,index_col=0)

  usr[fnum]=(len(df['InputID'].unique()))
  # usr[fnum]=(df['check'].mean())

usr = sorted(usr.items())
print(usr)


# %%
files = glob.glob('relation_first_check_??.csv')
for index,file in enumerate(files):
  fnum = re.sub(r"\D","",file)
  fnum = fnum[:2]
  print("fnum=",fnum)
  df=pd.read_csv(file,index_col=0,header=None).T
  df['lv']=df['q'].str[0]
  df['0']=df['0'].astype(float)
  df['1']=df['1'].astype(float)
  df=df.reindex(columns=['lv','jdg','q','0','1'])
  # print(df.dtypes)
  # print(df.groupby(['lv','jdg']).get_group(('1','f')))
  print(df)
  print(df['lv'].value_counts(normalize=True))
  df[['jdg','0','1','lv']].groupby(['lv','jdg']).mean().T.to_csv('relation_first_check_mean_'+str(fnum)+'.csv')

# %%
files = glob.glob('relation_first_check_??.csv')
for index,file in enumerate(files):
  fnum = re.sub(r"\D","",file)
  fnum = fnum[:2]
  print("fnum=",fnum)
  df=pd.read_csv(file,index_col=0,header=None).T
  df['lv']=df['q'].str[0]
  df['0']=df['0'].astype(float)
  df['1']=df['1'].astype(float)
  df=df.reindex(columns=['lv','jdg','q','0','1'])
  # print(df.dtypes)
  # print(df.groupby(['lv','jdg']).get_group(('1','f')))
  df['lv']=df['lv']+'_'+df['jdg']
  print(df)
  print(df['lv'].value_counts(normalize=True))
  df[['jdg','0','1','lv']].groupby(['lv',]).sum().T.to_csv('relation_first_check_ratio_'+str(fnum)+'.csv')

# %%
# !zip -r relation_strategy.zip /content

# from google.colab import files

# files.download("relation_strategy.zip")

# %%
files = glob.glob('monsakun_log_*_check.csv')
for index,file in enumerate(files):
  fnum = re.sub(r"\D","",file)
  fnum = fnum[:2]
  print("fnum=",fnum)

  fname = 'monsakun_log_02_check.csv'
  df = pd.read_csv(file,index_col=0)
  df['relation_st'] = 0

  #関係文の定義
  if fnum == '02' or fnum == '05':
    df.loc[df['slot1']==3,'relation_st'] = 'card_3'
    df.loc[df['slot1']==5,'relation_st'] = 'card_5'
  else:
    df.loc[df['slot1']==2,'relation_st'] = 'card_2'
    df.loc[df['slot1']==5,'relation_st'] = 'card_5'

  #新しい列の作成（レベル＿問題番号）
  df['q']=df['lv'].astype(str).str.cat(df['asg'].astype(str),sep='_')

  #チェック数１での結果
  df2 = df[df['check']==1]

  pd.crosstab(df2['relation_st'],[df2['q'],df2['jdg']]).to_csv('relation_first_check_'+str(fnum)+'.csv')

  #戦略ごとのチェック数
  df2 = df[df['jdg']=='s']
  # if index == 0:
  df2.to_csv('relation_st'+str(fnum)+'.csv')


  pd.pivot_table(df2,index='relation_st',columns=['q'],values='check',aggfunc=[np.mean,len]).to_csv('strategy_check_'+str(fnum)+'.csv')
  pd.pivot_table(df2,index='relation_st',columns=['q'],values='stp',aggfunc=[np.mean,len]).to_csv('strategy_step_'+str(fnum)+'.csv')

# %%



