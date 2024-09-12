import numpy as np
import sympy as sp

def ZCB_return(R):
    return 1/R

def ZCB_func(T):
    pass

def expected_coupon(t, T):
    return ZCB_func(T)/ZCB_func(t)

def fwd_p(R, S_0):
    return R*S_0

def riskless_APR(T, r):
    return np.exp(r*T)

def riskless_air(R, T):
    return np.log(R)/T

def present_value_discrete(X, ZCBs):
    return np.dot(X, ZCBs)

def expected_value(S_T, P_S_T):
    # assert arrays
    return np.dot(S_T, P_S_T)

##############################
# IMPLEMENT VARIANCE IN S(T) #
##############################

def fwd_payout(S, K, long = False):
    if long:
        return S-K
    else:
        return K-S
    
def euro_opt_payout(S, K, call = True):
    #[S(T) - K]^+
    if call:
        return max(0, S - K)
    else: 
        return max(0, K-S)
    
def calculate_mean(data):
    """Calculate the mean of a dataset."""
    return sum(data) / len(data)

def calculate_variance(data, sample=True):
    """
    Calculate the variance of a dataset.
    
    Parameters:
    data: list or array-like, the dataset
    sample: bool, if True, use sample variance (n-1), else use population variance (n)
    
    Returns:
    variance: float, the variance of the dataset
    """
    mean = calculate_mean(data)
    n = len(data)
    
    # Sum of squared differences from the mean
    squared_diffs = [(x - mean)**2 for x in data]
    
    # If sample variance, divide by (n-1), otherwise divide by n
    if sample:
        return sum(squared_diffs) / (n - 1)
    else:
        return sum(squared_diffs) / n

def hedge_port(H, A):
    return np.dot(H, A)

def call_put_parity(C_0, P_0, S_0, K, R):
    return C_0-P_0==S_0-K/R

