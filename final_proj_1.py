import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker

# Option 1: If you installed Seaborn, uncomment the next two lines
# import seaborn as sns
# plt.style.use('seaborn-darkgrid')

# Option 2: Use an alternative Matplotlib style
plt.style.use('ggplot')  # You can change this to any available style

###################################
# Black-Scholes & Implied Vol
###################################
def d1(S, K, r, q, sigma, T):
    return (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

def d2(S, K, r, q, sigma, T):
    return d1(S, K, r, q, sigma, T) - sigma*np.sqrt(T)

def bs_call_price(S, K, r, q, sigma, T):
    d_1 = d1(S, K, r, q, sigma, T)
    d_2 = d2(S, K, r, q, sigma, T)
    return S*np.exp(-q*T)*norm.cdf(d_1) - K*np.exp(-r*T)*norm.cdf(d_2)

def bs_put_price(S, K, r, q, sigma, T):
    d_1 = d1(S, K, r, q, sigma, T)
    d_2 = d2(S, K, r, q, sigma, T)
    return K*np.exp(-r*T)*norm.cdf(-d_2) - S*np.exp(-q*T)*norm.cdf(-d_1)

def implied_volatility(market_price, S, K, r, q, T, option_type='call'):
    def f(sigma):
        if option_type == 'call':
            return bs_call_price(S, K, r, q, sigma, T) - market_price
        else:
            return bs_put_price(S, K, r, q, sigma, T) - market_price
    
    try:
        iv = brentq(f, 1e-4, 3.0)
        return iv
    except:
        return np.nan

###################################
# Data from Hypothetical Scenario
###################################
S = 160.25    # Spot from Yahoo Finance (example)
q = 0.0355    # Approx. continuous dividend yield from annualized yield
r = 0.015     # Approx risk-free rate from 1-month T-Bill on this date
T = 40/365.0  # Time fraction for ~40 days to expiry

option_data = [
    # Format: (Strike, OptionType, MidPrice)
    (155, 'call', 7.30),
    (160, 'call', 4.60),
    (165, 'call', 2.40),
    (155, 'put',  2.70),
    (160, 'put',  5.10),
    (165, 'put',  8.25)
]

###################################
# Compute Implied Volatilities
###################################
iv_records = []
for strike, otype, mprice in option_data:
    iv = implied_volatility(mprice, S, strike, r, q, T, option_type=otype)
    iv_records.append({
        'Strike': strike,
        'Time_to_Expiry': T,
        'OptionType': otype,
        'MarketPrice': mprice,
        'ImpliedVol': iv
    })

implied_vol_df = pd.DataFrame(iv_records)
print("Computed Implied Volatilities:")
print(implied_vol_df)

###################################
# Plotting IV Surface
# Since we only have one maturity, let's simulate additional expiries for demonstration
###################################
# Assume additional expiries with slight variations in IV
additional_data = [
    {'Strike': 155, 'Time_to_Expiry':0.2, 'ImpliedVol':0.0922 + 0.01},
    {'Strike': 160, 'Time_to_Expiry':0.2, 'ImpliedVol':0.0912 + 0.01},
    {'Strike': 165, 'Time_to_Expiry':0.2, 'ImpliedVol':0.1068 + 0.01},
    {'Strike': 155, 'Time_to_Expiry':0.3, 'ImpliedVol':0.0922 + 0.02},
    {'Strike': 160, 'Time_to_Expiry':0.3, 'ImpliedVol':0.0912 + 0.02},
    {'Strike': 165, 'Time_to_Expiry':0.3, 'ImpliedVol':0.1068 + 0.02}
]

# Create DataFrame for surface
surface_data = pd.DataFrame([
    {'Strike': rec['Strike'], 'Time_to_Expiry': rec['Time_to_Expiry'], 'ImpliedVol': rec['ImpliedVol']}
    for rec in iv_records if rec['OptionType']=='call'  # Using calls for the surface
] + additional_data)

def plot_iv_surface(implied_vol_data):
    """
    Plot an implied volatility surface with enhanced aesthetics.
    expected columns: ['Strike', 'Time_to_Expiry', 'ImpliedVol']
    """
    pivot = implied_vol_data.pivot(index='Strike', columns='Time_to_Expiry', values='ImpliedVol')
    strikes = pivot.index.values
    expiries = pivot.columns.values
    X, Y = np.meshgrid(expiries, strikes)
    Z = pivot.values

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.cm.viridis
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.8)

    # Customize labels and title
    ax.set_xlabel('Time to Expiry (Years)', labelpad=15, fontsize=14, fontweight='bold')
    ax.set_ylabel('Strike Price (USD)', labelpad=15, fontsize=14, fontweight='bold')
    ax.set_zlabel('Implied Volatility (%)', labelpad=15, fontsize=14, fontweight='bold')
    ax.set_title('Implied Volatility Surface for CVX Stock Options', fontsize=16, fontweight='bold')

    # Customize ticks
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(0.02))
    ax.zaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    # Add color bar
    cb = fig.colorbar(surf, shrink=0.5, aspect=20)
    cb.set_label('Implied Volatility (%)', fontsize=14, fontweight='bold')

    # Improve the view angle
    ax.view_init(elev=30, azim=-60)

    plt.tight_layout()
    plt.show()

plot_iv_surface(surface_data)

###################################
# American Option Pricing with Dividends Example
###################################
def crr_american_option(S, K, r, q, sigma, T, M=100, option_type='call', dividends=[]):
    dt = T / M
    u = np.exp(sigma * np.sqrt(dt))
    d = 1/u
    p = (np.exp((r - q)*dt) - d) / (u - d)
    
    # Initialize stock price tree
    stock_prices = np.zeros((M+1, M+1))
    stock_prices[0,0] = S
    for i in range(1, M+1):
        stock_prices[i,0] = stock_prices[i-1,0]*u
        for j in range(1, i+1):
            stock_prices[i,j] = stock_prices[i-1,j-1]*d

    # Adjust for dividends
    for div_time, div_amount in dividends:
        step_ex = int(div_time / dt)
        if 0 < step_ex <= M:
            for j in range(step_ex+1):
                stock_prices[step_ex, j] = max(stock_prices[step_ex, j] - div_amount, 0)

    # Initialize option value at maturity
    option_values = np.zeros((M+1, M+1))
    if option_type == 'call':
        option_values[M,:] = np.maximum(stock_prices[M,:] - K, 0)
    else:
        option_values[M,:] = np.maximum(K - stock_prices[M,:], 0)
    
    # Backward induction
    for i in reversed(range(M)):
        for j in range(i+1):
            hold = np.exp(-r*dt) * (p * option_values[i+1,j] + (1 - p) * option_values[i+1,j+1])
            if option_type == 'call':
                exercise = max(stock_prices[i,j] - K, 0)
            else:
                exercise = max(K - stock_prices[i,j], 0)
            option_values[i,j] = max(hold, exercise)
    
    return option_values[0,0]

def plot_american_option_price_changes(S, K, r, q, sigma, T, dividends, option_type='call'):
    """
    Plot how American option prices change if the second dividend is increased by 20%.
    """
    base_price = crr_american_option(S, K, r, q, sigma, T, M=100, option_type=option_type, dividends=dividends)

    # Modify the second dividend
    mod_dividends = dividends.copy()
    if len(mod_dividends) > 1:
        mod_dividends[1] = (mod_dividends[1][0], mod_dividends[1][1] * 1.2)
    
    mod_price = crr_american_option(S, K, r, q, sigma, T, M=100, option_type=option_type, dividends=mod_dividends)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_labels = ['Base Dividend', 'Second Dividend +20%']
    prices = [base_price, mod_price]
    bar_colors = ['#1f77b4', '#ff7f0e']  # Blue and orange

    bars = ax.bar(bar_labels, prices, color=bar_colors, alpha=0.7, edgecolor='black')

    # Add title and labels
    ax.set_title(f'American {option_type.capitalize()} Option Price under Dividend Change\nCVX Stock', fontsize=16, fontweight='bold')
    ax.set_ylabel('Option Price (USD)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(prices) * 1.3)

    # Add gridlines
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

    # Annotate bars with prices
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'${height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

# Example: Assume we have two dividends expected before expiry:
# T=0.1096 years ~ 40 days. Suppose dividends fall at 0.04 years and 0.08 years with amount = $1.42
dividends_schedule = [(0.04, 1.42), (0.08, 1.42)]
plot_american_option_price_changes(S, 160, r, q, 0.20, T, dividends_schedule, option_type='call')
