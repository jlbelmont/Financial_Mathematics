import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import CubicSpline

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

#######################
# Greeks Calculations #
#######################

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

######################
# Least Squares Reg. #
######################

def least_squares_fit(x_data, y_data):
    """
    Find the least-squares best fit coefficients for the model:
    f(x) = p1 + p2 * exp(x) + p3 * exp(-x)
    
    Parameters:
    x_data (array-like): The x-values of the data points
    y_data (array-like): The y-values of the data points
    
    Returns:
    tuple: Coefficients (p1, p2, p3) that minimize the least-squares error
    """
    # Set up the matrix A where each row corresponds to [1, exp(x), exp(-x)] for each x in the data
    A = np.column_stack((np.ones_like(x_data), np.exp(x_data), np.exp(-x_data)))

    # Solve the least-squares problem: A @ p = y_data
    p, _, _, _ = np.linalg.lstsq(A, y_data, rcond=None)

    # Return the coefficients p1, p2, p3
    return p[0], p[1], p[2]

def plot_least_squares_fit(x_data, y_data, p1, p2, p3, num_points=81):
    """
    Plot the least-squares best fit function f(x) = p1 + p2 * exp(x) + p3 * exp(-x)
    along with the data points with improved aesthetics.
    
    Parameters:
    x_data (array-like): The x-values of the data points
    y_data (array-like): The y-values of the data points
    p1 (float): Coefficient for the constant term
    p2 (float): Coefficient for exp(x)
    p3 (float): Coefficient for exp(-x)
    num_points (int): Number of equispaced points for plotting the fitted function
    """
    # Generate equispaced points for plotting the function
    x_plot = np.linspace(min(x_data), max(x_data), num_points)

    # Calculate the corresponding y values for the fitted function f(x)
    y_plot = p1 + p2 * np.exp(x_plot) + p3 * np.exp(-x_plot)

    # Create a figure with a larger size
    plt.figure(figsize=(8, 6))

    # Plot the original data points with larger red markers
    plt.scatter(x_data, y_data, color='darkred', label='Data Points', zorder=5, s=100, edgecolor='black')

    # Plot the fitted function with a thicker line
    plt.plot(x_plot, y_plot, label=r'$f(x) = p_1 + p_2 e^x + p_3 e^{-x}$', color='royalblue', lw=2.5)

    # Add gridlines with a light style
    plt.grid(True, which='both', linestyle='--', lw=0.7, color='gray', alpha=0.7)

    # Add x and y axis labels with larger fonts
    plt.xlabel('x', fontsize=14, labelpad=10)
    plt.ylabel('f(x)', fontsize=14, labelpad=10)

    # Set a title with a larger font size
    plt.title('Least-Squares Best Fit Function', fontsize=16, pad=20)

    # Customize ticks for a cleaner look
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add a legend with a larger font size and edge
    plt.legend(fontsize=12, frameon=True, shadow=True)

    # Adjust the axis limits slightly for better spacing
    plt.xlim(min(x_data) - 0.5, max(x_data) + 0.5)
    plt.ylim(min(y_data) - 1, max(y_data) + 1)

    # Apply a function to format y-axis labels as integers
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda val, _: f'{int(val)}'))

    # Save the plot to a file (optional)
    # plt.savefig("least_squares_fit.png", dpi=300, bbox_inches='tight')

    # Show the plot
    plt.tight_layout()
    plt.show()

######################
# PREMIUM PREDICTION #
######################

def unweighted_quadratic_regression(strike_prices, premiums, target_strike):
    # Fit a quadratic polynomial (degree 2)
    coeffs = np.polyfit(strike_prices, premiums, 2)
    # Generate the quadratic function
    quadratic_func = np.poly1d(coeffs)
    # Estimate the premium for the target strike price
    estimated_premium = quadratic_func(target_strike)
    return estimated_premium

# def weighted_quadratic_regression(strike_prices, premiums, open_interest, target_strike):
#     # Fit a quadratic polynomial (degree 2) with weights
#     coeffs = np.polyfit(strike_prices, premiums, 2, w=open_interest)
#     # Generate the quadratic function
#     quadratic_func = np.poly1d(coeffs)
#     # Estimate the premium for the target strike price
#     estimated_premium = quadratic_func(target_strike)
#     return estimated_premium

def weighted_quadratic_regression(x, y, w, S0):
    # Define the design matrix F for quadratic regression
    F = np.vstack((np.ones(len(x)), x, x**2)).T  # Shape (n, 3)

    # Create the weight matrix W as a diagonal matrix
    W = np.diag(w)

    # Calculate the regression coefficients p using the weighted least squares formula
    p = np.linalg.inv(F.T @ W @ F) @ (F.T @ W @ y)

    # Estimate the premium at the spot price S0
    estimate = np.array([1, S0, S0**2]) @ p  # F(S0)

    return estimate

def polynomial_interpolation(strike_prices, premiums, target_strike):
    # Fit a polynomial of degree len(strike_prices)-1 (degree 6 here)
    coeffs = np.polyfit(strike_prices, premiums, len(strike_prices)-1)
    # Generate the polynomial function
    poly_func = np.poly1d(coeffs)
    # Estimate the premium for the target strike price
    estimated_premium = poly_func(target_strike)
    return estimated_premium

def spline_interpolation(strike_prices, premiums, target_strike):
    # Create a cubic spline interpolation
    spline_func = CubicSpline(strike_prices, premiums)
    # Estimate the premium for the target strike price
    estimated_premium = spline_func(target_strike)
    return estimated_premium