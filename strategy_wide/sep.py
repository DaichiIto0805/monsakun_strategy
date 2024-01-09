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

df = pd.read_csv("strategy_wide/cluster_strategy_wide_2_meaning.csv")
# %%
