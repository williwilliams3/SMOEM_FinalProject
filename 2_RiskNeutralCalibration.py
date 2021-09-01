#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 18:55:55 2021

@author: bwilliams

SMOEM 

Final Project ex2

Risk Neutral Calibration

"""

import os 
import pandas as pd
import numpy as np
from scipy.stats import norm, gamma
from scipy.integrate import quad
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
from datetime import datetime 

def load_process_data(work_dir):
    os.chdir(work_dir)
    dfopt_raw = pd.read_csv('data/Options_Prices_Calendar_2021.csv')
    dffwd_raw = pd.read_csv('data/Forward_Prices.csv',  dtype={'DeliveryPeriod': object})
    
    
    dfopt_raw['ExpiryDate'] = pd.to_datetime(dfopt_raw['ExpiryDate'])
    dffwd_raw['TradingDate'] = pd.to_datetime(dffwd_raw['TradingDate'])
    
    # Take the calendar Fwd
    df_opt = dfopt_raw[(dfopt_raw['Underlying'] == 'DEBY 2021.01')&(dfopt_raw.Type == 'C')]
    # Take the 2021
    df_fwd = dffwd_raw[(dffwd_raw.Contract == 'DEBY') & (dffwd_raw.DeliveryPeriod=='2021.01') ]
    
    # Remove where increment is zero to avoid errors
    
    return df_opt, df_fwd


def black_76_call( fs, k, t, r, V):
    # Inputs: option_type = "p" or "c", fs = price of underlying, x = strike, t = time to expiration, r = risk free rate
    #          v = implied volatility

    # t__sqrt = math.sqrt(t)
    # d1 = (math.log(fs / x) + (b + (v * v) / 2) * t) / (v * t__sqrt)
    # d2 = d1 - v * t__sqrt
    eps = np.finfo(float).eps
    
    if fs>np.finfo('float').max:
        fs = np.finfo('float').max
    
    
    if V < eps:
        d1 = np.inf
        d2 = np.inf
        

    else:
        d1 = (np.log(fs / k) + V/2) / np.sqrt(V)
        d2 = d1 - np.sqrt(V)
        
    value = fs * np.exp((- r) * t) * norm.cdf(d1) - k* np.exp(-r * t) * norm.cdf(d2)
    
    return value


# Model Calibration

# I need to implement call option pricing by integrating  BS-76 multiplied by gamma density 
# and then find parameters that minimize squared difference


def squared_loss_calibration(df_opt, df_fwd, params):

    F0 = float(df_fwd.SettlementPrice)
    r = 0.01
    t0 = df_fwd.TradingDate.values[0]
    maturities = df_opt.ExpiryDate
    T = (maturities - t0).dt.days/365
    K = df_opt.Strike
    P = df_opt.SettlementPrice    

    theta = params[0]
    nu = params[1]
    sigma = params[2]

    def integrator(g):
        Fg = F0*np.exp(omega*t + g*(theta+sigma**2/2))
        Vg = sigma**2*g
        
        gamma_pdf= gamma.pdf(g, a = t/nu, scale = nu)
        bs_76 = black_76_call( Fg, k, t, r, Vg)
        
        eps = sys.float_info.min
        
        if (gamma_pdf < eps) | (bs_76< eps):
            return 0
        
        if bs_76> sys.float_info.max:
             bs_76 = sys.float_info.max
        
        value = gamma_pdf * bs_76
        return value


    error = np.zeros(len(T))
    ci_ols = np.zeros(len(T))
    for i in range(len(T)):
        t = T.iloc[i]
        k = K.iloc[i]
        ci = P.iloc[i]    
        
        omega = (1/nu)*np.log(1-sigma**2*nu/2-theta*nu) 
        
        integral = quad(integrator, 0.0, np.inf)
        
        ci_theta = integral[0]
        
        error[i] = (ci - ci_theta)**2
        ci_ols[i] = ci_theta
    return error,  ci_ols 

def target_function(params):
    error, _ = squared_loss_calibration(df_opt, df_fwd, params)
    return np.sum(error)


def ols_estimation(df_opt, df_fwd, params, bnds, seed=42):
    vInicio = datetime.now()
    np.random.seed(seed)
    # , options = dict(maxiter = 10)
    output = minimize(target_function, params, bounds = bnds )                     
    vFin = datetime.now()
    print("--- %s seconds ---" % (vFin - vInicio).total_seconds())   
    print(output)
    return output['x']


def plot_prices(P_ols):
    
    # One plot for each maturity

    t0 = df_fwd.TradingDate.values[0]
    maturities = df_opt.ExpiryDate
    T = (maturities - t0).dt.days/365
    K = df_opt.Strike
    P = df_opt.SettlementPrice    
    dif_Ps = P_ols-P
    
    Tunique = np.unique(T)
    
    fig, axs = plt.subplots(2, 2, figsize=(15,15))
    
    
    axs[0, 0].set_title('Maturity  in years '+ str(round(Tunique[0],2)) )
    axs[0, 0].scatter(K[T==Tunique[0]], P_ols[T==Tunique[0]], label = 'OLS Call Prices', marker='x', alpha=0.5, s=5)
    axs[0, 0].scatter(K[T==Tunique[0]], P[T==Tunique[0]], label = 'Mkt Call Prices', alpha=0.5, s=5)
    axs[0, 0].set_xlabel('Strike')
    axs[0, 0].set_ylabel('Price')
    axs[0, 0].legend(loc='upper left')
    ax0 = axs[0, 0].twinx()
    ax0.scatter(K[T==Tunique[0]], dif_Ps[T==Tunique[0]], label = 'Difference', marker='x', alpha=0.5, s=5, color='green')
    ax0.set_ylim([-0.5, 2])
    ax0.legend(loc='upper right')
    
    axs[0, 1].set_title('Maturity  in years '+ str(round(Tunique[1],2)) )
    axs[0, 1].scatter(K[T==Tunique[1]], P_ols[T==Tunique[1]], label = 'OLS Call Prices', marker='x', alpha=0.5, s=5)
    axs[0, 1].scatter(K[T==Tunique[1]], P[T==Tunique[1]], label = 'Mkt Call Prices', alpha=0.5, s=5)
    axs[0, 1].set_xlabel('Strike')
    axs[0, 1].set_ylabel('Price')
    axs[0, 1].legend(loc='upper left')
    ax1 = axs[0, 1].twinx()
    ax1.scatter(K[T==Tunique[1]], dif_Ps[T==Tunique[1]], label = 'Difference', marker='x', alpha=0.5, s=5, color='green')
    ax1.set_ylim([-0.5, 2.5])
    ax1.legend(loc='upper right')
    
    axs[1, 0].set_title('Maturity  in years '+ str(round(Tunique[2],2)) )
    axs[1, 0].scatter(K[T==Tunique[2]], P_ols[T==Tunique[2]], label = 'OLS Call Prices', marker='x', alpha=0.5, s=5)
    axs[1, 0].scatter(K[T==Tunique[2]], P[T==Tunique[2]], label = 'Mkt Call Prices', alpha=0.5, s=5)
    axs[1, 0].set_xlabel('Strike')
    axs[1, 0].set_ylabel('Price')
    axs[1, 0].legend(loc='upper left')
    ax2 = axs[1, 0].twinx()
    ax2.scatter(K[T==Tunique[2]], dif_Ps[T==Tunique[2]], label = 'Difference', marker='x', alpha=0.5, s=5, color='green')
    ax2.set_ylim([-1, 4])
    ax2.legend(loc='upper right')
    
    axs[1, 1].set_title('Maturity  in years '+ str(round(Tunique[3],2)) )
    axs[1, 1].scatter(K[T==Tunique[3]], P_ols[T==Tunique[3]], label = 'OLS Call Prices', marker='x', alpha=0.5, s=5)
    axs[1, 1].scatter(K[T==Tunique[3]], P[T==Tunique[3]], label = 'Mkt Call Prices', alpha=0.5, s=5)
    axs[1, 1].set_xlabel('Strike')
    axs[1, 1].set_ylabel('Price')
    axs[1, 1].legend(loc='upper left')
    ax3 = axs[1, 1].twinx()
    ax3.scatter(K[T==Tunique[3]], dif_Ps[T==Tunique[3]], label = 'Difference', marker='x', alpha=0.5, s=5, color='green')
    ax3.set_ylim([-1, 5])
    ax3.legend(loc='upper right')
    
    plt.show()


    

work_dir = '/Users/bwilliams/GoogleDrive/UniversityOfHelsinki/Spring2021/SMFOEM/Week7/Project_BW'
df_opt, df_fwd = load_process_data(work_dir)


params_mm = np.array( (-0.20328199035811578, 0.19394121669473757, 0.004539030408410662) )
param_matteo = np.array((1.05, 0.02, 0.2))
seed=42
bnds = ((-2.7, 2.7), (0.001, 0.8), (0.0001, 1.1) )
ols_params = ols_estimation(df_opt, df_fwd, params_mm, bnds)

error_ols, P_ols =  squared_loss_calibration(df_opt, df_fwd, ols_params)
plot_prices(P_ols)
