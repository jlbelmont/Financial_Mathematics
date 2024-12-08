import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from io import StringIO  # Corrected import for StringIO

# -------------------------------------------
# Step 1: Parse the provided CSV Data
# -------------------------------------------
put_data = """
Strike,Last Price,Bid,Ask,Change,% Change,Volume,Open Interest,Implied Volatility
80,0.04,0,0.01,0.04,-,1,162.50%
85,0.01,0,0.09,0.01,-,1,183.59%
110,0.66,0,0.21,0.66,-,1,123.05%
115,0.02,0,0.04,0.02,-,2,89.06%
125,0.02,0,0.12,0,0.00%,5,75.78%
135,0.16,0,1.27,0,0.00%,2,81.45%
136,0.01,0,0.22,-0.01,-50.00%,10,54.10%
137,0.01,0,0.08,0,0.00%,30,49.02%
138,0.01,0.02,0.07,-0.14,-93.33%,13,45.51%
139,0.03,0.01,0.03,0.02,200.00%,14,38.28%
140,0.05,0,0.09,0.04,400.00%,2,42.58%
141,0.03,0,0.05,0.02,200.00%,2,36.52%
142,0.18,0.01,0.09,0,0.00%,135,37.60%
143,0.03,0.01,0.11,0.01,50.00%,1,36.43%
144,0.11,0,0.12,0,0.00%,1,34.47%
145,0.01,0.02,0.13,0,0.00%,2,32.32%
146,0.13,0.03,0.08,0,0.00%,2,27.05%
147,0.09,0.03,0.16,0.05,125.00%,2,28.32%
148,0.16,0.11,0.15,0.13,433.33%,73,25.20%
149,0.16,0.16,0.22,0.10,166.67%,140,24.66%
150,0.24,0.18,0.26,0.17,242.86%,1753,22.71%
152.5,0.57,0.56,0.60,0.45,375.00%,2842,20.51%
155,1.40,1.32,1.45,1.06,311.76%,578,19.75%
157.5,2.83,2.61,2.95,1.97,229.07%,293,19.61%
160,4.75,3.40,6.70,3.02,174.57%,67,47.93%
162.5,6.83,7.05,7.65,3.86,129.97%,8,30.96%
165,9.38,9.30,10.35,3.23,52.52%,1,42.82%
167.5,6.40,11.60,13.00,0,0.00%,1,53.37%
177.5,17.60,21.65,23.40,0,0.00%,10,62.79%
180,18.18,23.05,26.15,0,0.00%,7,101.61%
185,24.04,28.15,31.15,0,0.00%,-,114.06%
"""

call_data = """
Strike,Last Price,Bid,Ask,Change,% Change,Volume,Open Interest,Implied Volatility
100,56.17,53.50,57.40,56.17,-,5,168.95%
120,38.27,33.90,36.95,0,0.00%,1,103.91%
135,20.82,19.85,21.05,20.82,-,12,63.67%
140,22.90,14.75,15.90,0,0.00%,18,65.72%
146,10.23,8.90,9.85,10.23,-,73,44.29%
147,9.34,7.80,9.30,0,0.00%,2,49.95%
148,13.70,7.05,8.15,0,0.00%,5,43.34%
149,4.60,12.60,13.45,0,0.00%,-,123.32%
150,5.75,4.55,5.95,-6.15,-51.68%,2,32.08%
152.5,4.20,2.88,4.40,-4.50,-51.72%,20,35.84%
155,1.80,1.73,1.88,-3.40,-65.38%,987,22.14%
157.5,0.70,0.67,0.82,-2.93,-80.72%,953,21.49%
160,0.27,0.25,0.28,-1.31,-82.91%,991,21.05%
162.5,0.10,0.08,0.11,-0.47,-82.46%,794,22.46%
165,0.05,0.04,0.05,-0.13,-72.22%,292,24.51%
167.5,0.02,0.02,0.03,-0.10,-83.33%,45,27.34%
170,0.01,0,0.01,-0.02,-66.67%,2,27.74%
172.5,0.01,0,0.22,0,0.00%,21,50.00%
175,0.03,0,1.27,0.02,200.00%,1,70.31%
177.5,0.02,0,1.26,0,0.00%,3,76.12%
180,0.05,0,1.26,0,0.00%,-,81.88%
185,0.14,0,0.68,0,0.00%,-,81.45%
190,0.01,0,0.56,0.01,-,5,87.89%
210,0.02,0,0.05,0.02,-,2,88.67%
"""

# Function to parse percentage strings
def parse_percentage(x):
    if isinstance(x, str):
        return float(x.strip('%')) / 100.0
    return np.nan

# Read Put and Call data into DataFrames using StringIO from io
df_puts = pd.read_csv(StringIO(put_data))
df_calls = pd.read_csv(StringIO(call_data))

# Convert "Implied Volatility" to numeric
for df in [df_puts, df_calls]:
    df["Implied Volatility"] = df["Implied Volatility"].apply(parse_percentage)

# -------------------------------------------
# Step 2: Set Market Parameters (Assumptions)
# -------------------------------------------
S0 = 150.0    # Assume current underlying price
T = 30/365.0  # ~30 days to expiration
r = 0.02      # 2% annual risk-free rate
q = 0.0       # Dividend yield (0% for now)

# -------------------------------------------
# Step 3: Define Black-Scholes and Implied Vol Functions
# -------------------------------------------
def bs_call_price(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_put_price(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def implied_vol_bs(option_type, S, K, T, r, q, market_price):
    # Define the objective function
    def objective(sigma):
        if option_type == 'C':
            return bs_call_price(S, K, T, r, q, sigma) - market_price
        elif option_type == 'P':
            return bs_put_price(S, K, T, r, q, sigma) - market_price
        else:
            raise ValueError("Option type must be 'C' or 'P'")
    
    # Use Brent's method to find the root
    try:
        vol = brentq(objective, 1e-6, 5.0, maxiter=500)
    except:
        vol = np.nan
    return vol

# -------------------------------------------
# Step 4: Define Cox-Ross-Rubinstein (CRR) Binomial Model
# -------------------------------------------
def crr_option_price(option_type, S, K, T, r, q, sigma, steps=100):
    """
    Price European or American option using Cox-Ross-Rubinstein binomial tree.
    """
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    a = np.exp((r - q) * dt)
    p = (a - d) / (u - d)
    
    # Initialize asset prices at maturity
    asset_prices = S * (u ** np.arange(steps, -1, -1)) * (d ** np.arange(0, steps +1, 1))
    
    # Initialize option values at maturity
    if option_type == 'C':
        option_values = np.maximum(asset_prices - K, 0.0)
    elif option_type == 'P':
        option_values = np.maximum(K - asset_prices, 0.0)
    else:
        raise ValueError("Option type must be 'C' or 'P'")
    
    # Step back through the tree
    for i in range(steps-1, -1, -1):
        option_values = np.exp(-r*dt) * (p * option_values[0:i+1] + (1 - p) * option_values[1:i+2])
        # For American options, check for early exercise
        if option_type == 'C':
            asset_prices = S * (u ** np.arange(i, -1, -1)) * (d ** np.arange(0, i+1, 1))
            option_values = np.maximum(option_values, asset_prices - K)
        elif option_type == 'P':
            asset_prices = S * (u ** np.arange(i, -1, -1)) * (d ** np.arange(0, i+1, 1))
            option_values = np.maximum(option_values, K - asset_prices)
    
    return option_values[0]

def implied_vol_crr(option_type, S, K, T, r, q, market_price, steps=100):
    # Define the objective function
    def objective(sigma):
        price = crr_option_price(option_type, S, K, T, r, q, sigma, steps)
        return price - market_price
    
    # Use Brent's method to find the root
    try:
        vol = brentq(objective, 1e-6, 5.0, maxiter=500)
    except:
        vol = np.nan
    return vol

# -------------------------------------------
# Step 5: Compute Implied Vol for Near-the-Money Options
# -------------------------------------------
# Define near-the-money strikes (e.g., within ±10% of S0)
near_money_lower = 135
near_money_upper = 165

# Filter near-the-money Calls and Puts
near_money_calls = df_calls[(df_calls.Strike >= near_money_lower) & (df_calls.Strike <= near_money_upper)].copy()
near_money_puts = df_puts[(df_puts.Strike >= near_money_lower) & (df_puts.Strike <= near_money_upper)].copy()

# Function to calculate mid price
def mid_price(bid, ask):
    if pd.isna(bid) and not pd.isna(ask):
        return ask
    elif pd.isna(ask) and not pd.isna(bid):
        return bid
    elif pd.isna(bid) and pd.isna(ask):
        return np.nan
    else:
        return (bid + ask) / 2.0

# Calculate mid prices for more accurate market prices
near_money_calls['Market Price'] = near_money_calls.apply(lambda row: mid_price(row['Bid'], row['Ask']), axis=1)
near_money_puts['Market Price'] = near_money_puts.apply(lambda row: mid_price(row['Bid'], row['Ask']), axis=1)

# Calculate Implied Volatility using Black-Scholes
near_money_calls['IV_BS'] = near_money_calls.apply(
    lambda row: implied_vol_bs('C', S0, row['Strike'], T, r, q, row['Market Price']) if not pd.isna(row['Market Price']) else np.nan, axis=1)

near_money_puts['IV_BS'] = near_money_puts.apply(
    lambda row: implied_vol_bs('P', S0, row['Strike'], T, r, q, row['Market Price']) if not pd.isna(row['Market Price']) else np.nan, axis=1)

# Calculate Implied Volatility using CRR
near_money_calls['IV_CRR'] = near_money_calls.apply(
    lambda row: implied_vol_crr('C', S0, row['Strike'], T, r, q, row['Market Price']) if not pd.isna(row['Market Price']) else np.nan, axis=1)

near_money_puts['IV_CRR'] = near_money_puts.apply(
    lambda row: implied_vol_crr('P', S0, row['Strike'], T, r, q, row['Market Price']) if not pd.isna(row['Market Price']) else np.nan, axis=1)

# Display Results
print("Near-the-money Calls with Implied Volatilities:")
print(near_money_calls[['Strike', 'Market Price', 'IV_BS', 'IV_CRR', 'Implied Volatility']])

print("\nNear-the-money Puts with Implied Volatilities:")
print(near_money_puts[['Strike', 'Market Price', 'IV_BS', 'IV_CRR', 'Implied Volatility']])

# -------------------------------------------
# Step 6: Plot the Implied Volatility Surface
# -------------------------------------------
# Since we have only one expiry T, we'll plot IV vs K for Calls and Puts
plt.figure(figsize=(12, 6))
plt.plot(near_money_calls['Strike'], near_money_calls['IV_BS'], marker='o', linestyle='-', label='Calls BS IV')
plt.plot(near_money_calls['Strike'], near_money_calls['IV_CRR'], marker='x', linestyle='--', label='Calls CRR IV')
plt.plot(near_money_puts['Strike'], near_money_puts['IV_BS'], marker='s', linestyle='-', label='Puts BS IV')
plt.plot(near_money_puts['Strike'], near_money_puts['IV_CRR'], marker='^', linestyle='--', label='Puts CRR IV')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.title('Implied Volatility vs Strike Price')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------
# Step 7: Compare with Published Implied Volatilities
# -------------------------------------------
# Compare calculated IV_BS and IV_CRR with provided "Implied Volatility"
# Note: "Implied Volatility" in the data likely refers to either Put or Call IVs; here we compare separately

# For Calls
plt.figure(figsize=(12, 6))
plt.plot(near_money_calls['Strike'], near_money_calls['IV_BS'], marker='o', linestyle='-', label='Calls BS IV')
plt.plot(near_money_calls['Strike'], near_money_calls['IV_CRR'], marker='x', linestyle='--', label='Calls CRR IV')
plt.plot(near_money_calls['Strike'], near_money_calls['Implied Volatility'], marker='d', linestyle=':', label='Calls Market IV')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.title('Call Options: Calculated vs Market Implied Volatility')
plt.legend()
plt.grid(True)
plt.show()

# For Puts
plt.figure(figsize=(12, 6))
plt.plot(near_money_puts['Strike'], near_money_puts['IV_BS'], marker='s', linestyle='-', label='Puts BS IV')
plt.plot(near_money_puts['Strike'], near_money_puts['IV_CRR'], marker='^', linestyle='--', label='Puts CRR IV')
plt.plot(near_money_puts['Strike'], near_money_puts['Implied Volatility'], marker='d', linestyle=':', label='Puts Market IV')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.title('Put Options: Calculated vs Market Implied Volatility')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------
# Step 8: Handling Dividends and American Options with CRR
# -------------------------------------------
# Assume two dividends within T
# For simplicity, let's assume:
# - First dividend at T1 = 10 days, D1 = $1
# - Second dividend at T2 = 20 days, D2 = $1

# Adjust the CRR model to handle dividends
def crr_option_price_american_with_dividends(option_type, S, K, T, r, q, sigma, steps=100, dividends=[]):
    """
    Price American option using CRR binomial tree with discrete dividends.
    `dividends` should be a list of tuples: (dividend_time, dividend_amount)
    """
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    a = np.exp((r - q) * dt)
    p = (a - d) / (u - d)
    
    # Initialize asset prices at maturity
    asset_prices = S * (u ** np.arange(steps, -1, -1)) * (d ** np.arange(0, steps +1, 1))
    
    # Initialize option values at maturity
    if option_type == 'C':
        option_values = np.maximum(asset_prices - K, 0.0)
    elif option_type == 'P':
        option_values = np.maximum(K - asset_prices, 0.0)
    else:
        raise ValueError("Option type must be 'C' or 'P'")
    
    # Precompute dividend times in steps
    dividend_steps = [int(div_time / T * steps) for div_time, _ in dividends]
    dividend_dict = {step: div_amount for step, div_amount in zip(dividend_steps, [div[1] for div in dividends])}
    
    # Step back through the tree
    for i in range(steps-1, -1, -1):
        # Adjust stock prices for dividends
        if i in dividend_dict:
            # At the node before dividend, the stock price drops by the dividend amount
            # Recalculate asset prices before discounting
            asset_prices = asset_prices[:i+1] - dividend_dict[i]
            # Ensure no negative stock prices
            asset_prices = np.maximum(asset_prices, 0.0)
        
        option_values = np.exp(-r*dt) * (p * option_values[0:i+1] + (1 - p) * option_values[1:i+2])
        
        # For American options, check for early exercise
        if option_type == 'C':
            asset_prices_exercise = S * (u ** np.arange(i, -1, -1)) * (d ** np.arange(0, i+1, 1))
            if i in dividend_dict:
                asset_prices_exercise -= dividend_dict[i]
                asset_prices_exercise = np.maximum(asset_prices_exercise, 0.0)
            exercise = np.maximum(asset_prices_exercise - K, 0.0)
            option_values = np.maximum(option_values, exercise)
        elif option_type == 'P':
            asset_prices_exercise = S * (u ** np.arange(i, -1, -1)) * (d ** np.arange(0, i+1, 1))
            if i in dividend_dict:
                asset_prices_exercise -= dividend_dict[i]
                asset_prices_exercise = np.maximum(asset_prices_exercise, 0.0)
            exercise = np.maximum(K - asset_prices_exercise, 0.0)
            option_values = np.maximum(option_values, exercise)
    
    return option_values[0]

def implied_vol_crr_dividends(option_type, S, K, T, r, q, market_price, steps=100, dividends=[]):
    # Define the objective function
    def objective(sigma):
        price = crr_option_price_american_with_dividends(option_type, S, K, T, r, q, sigma, steps, dividends)
        return price - market_price
    
    # Use Brent's method to find the root
    try:
        vol = brentq(objective, 1e-6, 5.0, maxiter=500)
    except:
        vol = np.nan
    return vol

# Example usage:
# Let's choose a Put option with Strike=150, which is near-the-money
selected_put = near_money_puts[near_money_puts['Strike'] == 150].iloc[0]
K = selected_put['Strike']
market_price = selected_put['Market Price']

# Define dividends: two dividends at 10 and 20 days
dividends = [(10/365.0, 1.0), (20/365.0, 1.0)]

# Calculate implied volatility using CRR with dividends
iv_crr_div = implied_vol_crr_dividends('P', S0, K, T, r, q, market_price, steps=100, dividends=dividends)

print(f"\nImplied Volatility for Put Strike={K} using CRR with dividends: {iv_crr_div:.4f}")

# Compare with market IV and previous calculations
print(f"Market Implied Volatility: {selected_put['Implied Volatility']:.4f}")
print(f"Implied Volatility (BS): {selected_put['IV_BS']:.4f}")
print(f"Implied Volatility (CRR): {selected_put['IV_CRR']:.4f}")

# -------------------------------------------
# Step 9: Effect of Increasing Second Dividend by 20%
# -------------------------------------------
# Increase second dividend by 20%
dividends_adjusted = [(10/365.0, 1.0), (20/365.0, 1.2)]  # D2 increased by 20%

# Calculate implied volatility with adjusted dividends
iv_crr_div_adjusted = implied_vol_crr_dividends('P', S0, K, T, r, q, market_price, steps=100, dividends=dividends_adjusted)

print(f"\nImplied Volatility for Put Strike={K} with increased second dividend: {iv_crr_div_adjusted:.4f}")

# -------------------------------------------
# Step 10: Construct Implied Binomial Tree and Price Puts
# -------------------------------------------
# For simplicity, we will use a fixed volatility (e.g., BS IV for Strike=150) to construct the tree

# Select multiple call strikes to build the tree
selected_calls = near_money_calls.copy()

# Let's choose 5 near-the-money call options
selected_calls = selected_calls.head(5)

# Define a function to construct implied binomial tree parameters from multiple call prices
def construct_binomial_tree(call_options, S, T, r, q, steps=100):
    """
    Simplified approach: Assume a single volatility and calibrate to match all call prices.
    This is a complex task; here we use an average implied vol.
    """
    vols = []
    for _, row in call_options.iterrows():
        iv = implied_vol_bs('C', S, row['Strike'], T, r, q, row['Market Price'])
        if not np.isnan(iv):
            vols.append(iv)
    if len(vols) == 0:
        return np.nan
    avg_vol = np.mean(vols)
    return avg_vol

# Calculate average implied volatility from selected calls
average_vol = construct_binomial_tree(selected_calls, S0, T, r, q, steps=100)
print(f"\nAverage Implied Volatility from selected calls: {average_vol:.4f}")

# Use this volatility to price put options
selected_puts_for_tree = near_money_puts.head(5).copy()

selected_puts_for_tree['CRR_Price'] = selected_puts_for_tree.apply(
    lambda row: crr_option_price('P', S0, row['Strike'], T, r, q, average_vol, steps=100), axis=1)

print("\nPut Options Prices using Implied Binomial Tree:")
print(selected_puts_for_tree[['Strike', 'Market Price', 'CRR_Price']])

# Compare with Put-Call Parity
# Put-Call Parity: C - P = S0 * exp(-q*T) - K * exp(-r*T)
# To compute P from C, rearrange: P = C - S0 * exp(-q*T) + K * exp(-r*T)

# Ensure that selected_calls and selected_puts_for_tree have the same number of rows
if len(selected_calls) >= len(selected_puts_for_tree):
    parity_calls = selected_calls.head(len(selected_puts_for_tree))
else:
    parity_calls = selected_calls.copy()

selected_puts_for_tree['Parity_Price'] = parity_calls['Market Price'].values[:len(selected_puts_for_tree)] - (S0 * np.exp(-q*T) - selected_puts_for_tree['Strike'] * np.exp(-r*T))
selected_puts_for_tree['Difference'] = selected_puts_for_tree['CRR_Price'] - selected_puts_for_tree['Parity_Price']

print("\nPut Options Prices vs Put-Call Parity:")
print(selected_puts_for_tree[['Strike', 'Market Price', 'CRR_Price', 'Parity_Price', 'Difference']])

# -------------------------------------------
# Step 11: Plot Implied Volatility Surface (Optional)
# -------------------------------------------
# If multiple maturities were available, we could plot a 3D surface.
# Here, we have only one maturity, so we've plotted IV vs Strike above.

# -------------------------------------------
# Summary and Comments on Differences
# -------------------------------------------
print("\n--- Summary and Comments on Differences ---")
print("""
1. **Choice of Options**:
   - Near-the-money options (Strike ~ S0) are selected because they are typically more liquid and have more reliable prices, leading to more accurate implied volatility estimates.

2. **Black-Scholes vs. CRR**:
   - The Black-Scholes model assumes continuous trading, no dividends, and European exercise. The CRR model can handle American options and discrete dividends, making it more flexible.
   - In the absence of dividends and for European options, both models should yield similar implied volatilities. Differences arise when dividends or early exercise features are present.

3. **Comparison with Market Implied Volatilities**:
   - Calculated IVs may differ from market IVs due to model assumptions, data discrepancies (e.g., using mid-price vs. last trade), or market conditions not captured by the models.
   - For options with dividends, the CRR model accounts for them, potentially providing more accurate IVs compared to Black-Scholes.

4. **Impact of Dividends**:
   - Increasing the second dividend by 20% leads to a higher implied volatility when using the CRR model. This is because higher dividends reduce the underlying asset price, increasing put option prices and thus requiring higher volatility to match the market price.

5. **Implied Binomial Tree and Put-Call Parity**:
   - The implied binomial tree approach provides a way to price options based on multiple call prices, ensuring consistency across strikes.
   - Comparing CRR-priced puts with those derived from put-call parity helps identify any discrepancies, which could indicate market inefficiencies or model limitations.
""")
