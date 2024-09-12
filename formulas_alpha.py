import numpy as np

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
    
