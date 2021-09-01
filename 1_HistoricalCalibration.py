#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 15:25:29 2021

@author: bwilliams

SMOEM 

Final Project ex1

Historical Calibration

"""


import os 
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.optimize import minimize
from scipy.special import gamma, kv 
from  numpy import pi
import matplotlib.pyplot as plt


def load_process_data(work_dir):
    os.chdir(work_dir)
    df_raw = pd.read_csv('data/Historical_Prices_FWD_Germany.csv')
    df_raw['date'] = pd.to_datetime(df_raw['Data'])
    df_GF = df_raw[df_raw['date']<='2019-11-19'][['date','DEBY2021']]    
    df_GF['log_ret'] = np.log(df_GF['DEBY2021']) - np.log(df_GF['DEBY2021'].shift(1))   
    # Remove where increment is zero to avoid errors
    df =df_GF[(~df_GF['log_ret'].isna()) & (df_GF['log_ret']!=0)]
    return df


def momment_matching_vg_params(data, dt):
    # M = np.mean(data);
    # Matlab uses biased estimator
    V = np.var(data, ddof=1);
    S = skew(data);
    # scipy substracts 3 (the normal distr kurtosis ) by default
    K = kurtosis(data) + 3;
    sigma = np.sqrt(V/dt);
    nu = (K/3 -1)*dt;
    theta = (S* sigma * np.sqrt(dt))/(3* nu );
    return theta, sigma, nu


def VGdensity_2(x, theta, nu, sigma, T):
    v1 = 2* np.exp(( theta*(x))/sigma**2) / ( (nu**(T/nu)) * np.sqrt(2*pi) * sigma * gamma(T/nu) );
    M2 = (2* sigma**2)/ nu + theta**2;
    v3 = np.abs(x)/ np.sqrt(M2 );   
    v4 = v3**(T/nu - 0.5) ;
    v6 = (np.abs(x)* np.sqrt(M2))/ sigma**2;
    K = kv(T/nu - 0.5 , v6 );
    fx = v1 * v4 *K;
    return fx



def mle_estimation(data, params, seed=42):
    np.random.seed(seed)
    # maximum likelihood estimation    
    neg_loglikelihood = lambda params: -np.sum(np.log(VGdensity_2(data, params[0], params[1], params[2], dt)))
    output = minimize(neg_loglikelihood, params, method = 'Nelder-Mead' )
    print(output)
    return output['x']


def plot_mle_fit():
    x = np.linspace(np.min(data), np.max(data), 500)
    fx_st = VGdensity_2(x, theta, nu, sigma, dt)
    fx_mle = VGdensity_2(x, mle_params[0], mle_params[1], mle_params[2], dt)
    
    
    fig, ax1 = plt.subplots()
    ax1.set_ylim([0, 110])
    plt.hist(data, bins = 50, edgecolor='black', alpha=0.3, color='green', label='Market data')
    plt.legend(loc='best')
    
    ax2 = ax1.twinx() 
    ax2.set_ylim([0, 70])
    plt.plot(x, fx_st, color='red', label='Method of Momments')
    plt.plot(x, fx_mle, color='green', label='MLE fit')
    # fig.tight_layout() 
    plt.legend(loc='best')
    plt.show()
    
# Data Analysis 
def plot_data():
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(df['date'], df['DEBY2021'], label='German Forward Prices')
    ax1.set_title('German Forward Prices')
    ax2.plot(df['date'], df['log_ret'], label='Log Returns', color='red', alpha=0.5)
    ax2.set_title('Log Returns')
    plt.xticks(rotation=90)
    plt.show()
    



work_dir = '/Users/bwilliams/GoogleDrive/UniversityOfHelsinki/Spring2021/SMFOEM/Week7/Project_BW'
df = load_process_data(work_dir)

dt = 1/252;
data = df['log_ret']


# Basic statistics log returns    
M = np.mean(data);
# Matlab uses biased estimator
V = np.var(data, ddof=1);
S = skew(data);
# scipy substracts 3 (the normal distr kurtosis ) by default
K = kurtosis(data) + 3;

print('Mean: ',M)
print('Variance: ',M)
print('Skewness: ',S)
print('Kurtosis: ',K)

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.gofplots import qqplot
plot_pacf(data, lags=50)
plt.show()

qqplot(data, line = 's', alpha=0.3)
plt.show()

# MLE estimation
theta, sigma, nu = momment_matching_vg_params(data, dt)
params = np.array([theta, nu, sigma])
mle_params = mle_estimation(data, params)

print('Method of momments parameters: ', params)
print('MLE fit parameters: ', mle_params)

plot_mle_fit()












