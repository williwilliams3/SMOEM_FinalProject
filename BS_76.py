
import numpy as np
from scipy.stats import norm

def black_76_call( fs, k, t, r, V):
    # Inputs: option_type = "p" or "c", fs = price of underlying, x = strike, t = time to expiration, r = risk free rate
    #          v = implied volatility

    # t__sqrt = math.sqrt(t)
    # d1 = (math.log(fs / x) + (b + (v * v) / 2) * t) / (v * t__sqrt)
    # d2 = d1 - v * t__sqrt


    d1 = (np.log(fs / k) + V/2) / np.sqrt(V)
    d2 = d1 - np.sqrt(V)
    
    value = fs * np.exp((- r) * t) * norm.cdf(d1) - k* np.exp(-r * t) * norm.cdf(d2)
    
    return value
   
black_76_call( fs=102, k=100, t=2, r=0.05, V= 2* (0.25)**2)
# Corect value 13.748035669879897    
    
    
        
        