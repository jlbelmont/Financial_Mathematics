import numpy as np
import scipy.stats as si

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
        return S - K
    else:
        return K - S
    
def euro_opt_payout(S, K, call=True):
    """[S(T) - K]^+ for a call, [K - S(T)]^+ for a put."""
    if call:
        return max(0, S - K)
    else: 
        return max(0, K - S)

def calculate_mean(data):
    """Calculate the mean of a dataset."""
    return sum(data) / len(data)

def calculate_variance(data, sample=True):
    """Calculate the variance of a dataset."""
    mean = calculate_mean(data)
    n = len(data)
    squared_diffs = [(x - mean)**2 for x in data]
    if sample:
        return sum(squared_diffs) / (n - 1)
    else:
        return sum(squared_diffs) / n

def hedge_port(H, A):
    return np.dot(H, A)

def call_put_parity(C_0, P_0, S_0, K, R):
    return C_0 - P_0 == S_0 - K / R

#########################
# Black-Scholes Formula #
#########################

def black_scholes(S, K, T, r, sigma, call=True):
    """
    Black-Scholes formula for European call or put option.
    
    Parameters:
    S: Current stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free interest rate
    sigma: Volatility of the stock
    call: True for call option, False for put option
    
    Returns:
    Price of the option
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if call:
        price = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    else:
        price = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0)
    
    return price

#########################
# Greeks Calculations    #
#########################

def delta(S, K, T, r, sigma, call=True):
    """
    Calculate the Delta of an option.
    Delta measures the sensitivity of the option price to changes in the underlying asset's price.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    if call:
        return si.norm.cdf(d1)
    else:
        return si.norm.cdf(d1) - 1

def gamma(S, K, T, r, sigma):
    """
    Calculate the Gamma of an option.
    Gamma measures the rate of change of Delta with respect to changes in the underlying price.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return si.norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    """
    Calculate the Vega of an option.
    Vega measures the sensitivity of the option price to changes in volatility.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * si.norm.pdf(d1) * np.sqrt(T)

def theta(S, K, T, r, sigma, call=True):
    """
    Calculate the Theta of an option.
    Theta measures the sensitivity of the option price to the passage of time.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if call:
        theta = (-S * si.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * si.norm.cdf(d2))
    else:
        theta = (-S * si.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * si.norm.cdf(-d2))
    
    return theta / 365  # Return per day theta

def rho(S, K, T, r, sigma, call=True):
    """
    Calculate the Rho of an option.
    Rho measures the sensitivity of the option price to changes in the risk-free interest rate.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if call:
        return K * T * np.exp(-r * T) * si.norm.cdf(d2)
    else:
        return -K * T * np.exp(-r * T) * si.norm.cdf(-d2)

##############################
# Test example with values   #
##############################

# Parameters
# S = 100  # Current stock price
# K = 100  # Strike price
# T = 1    # Time to maturity (1 year)
# r = 0.05  # Risk-free rate (5%)
# sigma = 0.2  # Volatility (20%)

# Calculate option prices and Greeks
# call_price = black_scholes(S, K, T, r, sigma, call=True)
# put_price = black_scholes(S, K, T, r, sigma, call=False)

# print(f"Call Price: {call_price}")
# print(f"Put Price: {put_price}")
# print(f"Delta (Call): {delta(S, K, T, r, sigma, call=True)}")
# print(f"Gamma: {gamma(S, K, T, r, sigma)}")
# print(f"Vega: {vega(S, K, T, r, sigma)}")
# print(f"Theta (Call): {theta(S, K, T, r, sigma, call=True)}")
# print(f"Rho (Call): {rho(S, K, T, r, sigma, call=True)}")
