
import pandas as pd
import numpy as np
from scipy.stats import norm, chi2, chi2_contingency


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
    np_data = np.array(table)

    # カイ二乗検定
    chi_sqared, chi_p_value, df, exp = chi2_contingency(np_data)
    if chi_p_value < p_value:
        print(f'カイ二乗検定：有意水準{p_value}で有意差あり。')
    else:
        print(f'カイ二乗検定：有意水準{p_value}で有意差なし。')

    # インデックスとカラム名
    index = table.index
    column = table.columns

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
                if np_data[i][j]>=exp[i][j]:
                    pairs.append((index[i], column[j],'▲'))
                elif np_data[i][j]<exp[i][j]:
                    pairs.append((index[i], column[j],'▽'))
    if chi_p_value < p_value:
        return pairs
    else :
        return 0

# %%
