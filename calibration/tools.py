from _plotly_future_ import v4_subplots
from plotly.subplots import make_subplots

import itertools

import pandas as pd
pd.set_option("display.max_rows", 1500)
pd.set_option("display.max_columns", 1000)
import numpy as np
import scipy as scp
import scipy.stats as ss
from scipy.stats import norm

import sklearn.metrics as sm

from scipy.integrate import quad
import matplotlib.pyplot as plt

import scipy.special as scps
from statsmodels.graphics.gofplots import qqplot
from scipy.linalg import cholesky
from functools import partial
from scipy.optimize import minimize, brute, fmin
from IPython.display import display
import sympy; sympy.init_printing()
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from plotly.offline import init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(precision=4, suppress=True)

import pickle 
def display_matrix(m):
    display(sympy.Matrix(m))

import pandas as pd
import numpy as np


def slice_data(data, date='20130515', cp_flag='C', moneyness_range=[0.8, 1.2], volume_q=0.9, verbose=0):
    D = data.date == date
    C = data.cp_flag == cp_flag
    data = data.loc[D & C]

    # Calculate the total trading volume
    total_volume = np.sum(data.volume)

    moneyness = (data.strike_price / 1000) / data.loc[:, 'Adj Close']
    M = (moneyness >= moneyness_range[0]) & (moneyness <= moneyness_range[1])
    data = data.loc[M, :]

    q = np.quantile(data.volume, volume_q)
    Q = data.volume > q
    data = data.loc[Q, :]

    sliced_volume = np.sum(data.volume)

    if verbose > 0:
        return data, total_volume, sliced_volume

    return data


def unpivot(frame):
    N, K = frame.shape
    data = {'Z': frame.to_numpy().ravel('F'),
            'Y': np.asarray(frame.columns).repeat(N),
            'X': np.tile(np.asarray(frame.index), K)}
    return pd.DataFrame(data, columns=['X', 'Y', 'Z'])


def plot_result(K, HS_price, target_price):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)

    ax.scatter(K,
               target_price, marker='o', color='b', label = 'target_price')

    ax.scatter(K,
               HS_price, marker='x', color='r', label = 'calibration result')

    ax.set(xlabel='strike (moneyness)', ylabel='Call Price',
           title='Call prices derived from implied volatilities on S&P 500 on 15 May 2013')

    ax.legend()

# Mayer Paper Data
BS_price_1 = '''
0.2000 0.2000 0.2005 0.2048 0.2151 0.2260 0.2372
0.1500 0.1502 0.1522 0.1579 0.1718 0.1852 0.1984
0.1004 0.1021 0.1050 0.1136 0.1315 0.1473 0.1624
0.0523 0.0566 0.0612 0.0735 0.0951 0.1130 0.1297
0.0305 0.0364 0.0421 0.0560 0.0788 0.0975 0.1147
0.0132 0.0199 0.0262 0.0407 0.0641 0.0831 0.1007
0.0038 0.0089 0.0144 0.0281 0.0509 0.0701 0.0878
0.0008 0.0033 0.0070 0.0184 0.0396 0.0583 0.0759
0.0001 0.0004 0.0015 0.0069 0.0224 0.0390 0.0555
0.0000 0.0000 0.0005 0.0025 0.0117 0.0249 0.0394
0.0000 0.0000 0.0001 0.0010 0.0059 0.0154 0.0272
'''
BS_price_1 = BS_price_1 .replace('\n',' ').split(' ')[1:-1]

BS_price_1 = pd.DataFrame(np.array(BS_price_1 ).reshape(11,7).T,
             index = [1/12,2/12,3/12,6/12,12/12,18/12,24/12] ,
   columns=[.8,.85,.9,.95,.975,1,1.025,1.05,1.1,1.15,1.2])


BS_price_1 = BS_price_1.astype('float')

BS_price_1 = unpivot(BS_price_1)
BS_price_1.columns = ['Tau', 'K', 'BS_price']
BS_price_1.loc[:,'S'] = 1
BS_price_1.loc[:,'r'] = 0


v0_1 = 0.013681                                        # spot variance
kappa_1 = 1.605179                                     # mean reversion coefficient
theta_1 = 0.053318                                     # long-term mean of the variance
rho_1 = -0.620100                                      # correlation coefficient
vol_vol_1 = 0.590506                                     # (Vol of Vol) - Volatility of instantaneous variance

alpha = 0.75
par_1 = v0_1, kappa_1, theta_1, rho_1, vol_vol_1

print(pd.DataFrame([34.98774564765]))