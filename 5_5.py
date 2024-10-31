import numpy as np
from itertools import product

def CRRparams(T, r, v, N):
    """
    Compute Cox-Ross-Rubinstein (CRR) parameters.
    
    Parameters:
    T (float): Expiration time in years.
    r (float): Risk-free interest rate per year.
    v (float): Volatility (>0).
    N (int): Number of steps.
    
    Returns:
    tuple: (pu, up, R)
    """
    dt = T / N  # time step
    R = np.exp(r * dt)  # discount factor per step
    up = np.exp(v * np.sqrt(dt))  # up-factor
    down = 1 / up  # down-factor
    pu = (np.exp(r * dt) - down) / (up - down)  # risk-neutral up probability
    return pu, up, R

def generate_all_paths(N):
    """
    Generate all possible paths for N steps in a binomial tree.
    
    Parameters:
    N (int): Number of steps.
    
    Returns:
    np.ndarray: Array of shape (2^N, N) with binary paths.
    """
    return np.array(list(product([0, 1], repeat=N)))

def compute_stock_prices(S0, up, down, paths):
    """
    Compute stock prices for all paths.
    
    Parameters:
    S0 (float): Initial stock price.
    up (float): Up-factor.
    down (float): Down-factor.
    paths (np.ndarray): Array of binary paths.
    
    Returns:
    np.ndarray: Array of shape (2^N, N+1) with stock prices for each path.
    """
    num_paths, N = paths.shape
    Sbar = np.zeros((num_paths, N + 1))
    Sbar[:, 0] = S0
    for step in range(1, N + 1):
        Sbar[:, step] = Sbar[:, step - 1] * np.where(paths[:, step - 1] == 1, up, down)
    return Sbar

def compute_averages(Sbar):
    """
    Compute the average stock price for each path.
    
    Parameters:
    Sbar (np.ndarray): Array of stock prices for each path.
    
    Returns:
    np.ndarray: Array of average stock prices.
    """
    return np.mean(Sbar, axis=1)

def compute_payoffs(Sbar, averages):
    """
    Compute call and put option payoffs.
    
    Parameters:
    Sbar (np.ndarray): Array of stock prices for each path.
    averages (np.ndarray): Array of average stock prices for each path.
    
    Returns:
    tuple: (call_payoffs, put_payoffs)
    """
    S_T = Sbar[:, -1]
    call_payoffs = np.maximum(S_T - averages, 0)
    put_payoffs = np.maximum(averages - S_T, 0)
    return call_payoffs, put_payoffs

def compute_path_probabilities(pu, N, paths):
    """
    Compute the probability of each path.
    
    Parameters:
    pu (float): Risk-neutral up probability.
    N (int): Number of steps.
    paths (np.ndarray): Array of binary paths.
    
    Returns:
    np.ndarray: Array of path probabilities.
    """
    # Count number of up moves in each path
    u_counts = np.sum(paths, axis=1)
    d_counts = N - u_counts
    # Compute probabilities
    return (pu ** u_counts) * ((1 - pu) ** d_counts)

def CRRfltAD(T, S0, r, v, N):
    """
    Price floating strike call and put options using the CRR model with Arrow-Debreu securities.
    
    Parameters:
    T (float): Expiration time in years.
    S0 (float): Initial stock price.
    r (float): Risk-free interest rate per year.
    v (float): Volatility (>0).
    N (int): Number of steps.
    
    Returns:
    tuple: (C0, P0) Option prices at time 0.
    """
    pu, up, R = CRRparams(T, r, v, N)
    down = 1 / up

    # Generate all possible paths
    paths = generate_all_paths(N)  # Shape: (2^N, N)

    # Compute stock prices for all paths
    Sbar = compute_stock_prices(S0, up, down, paths)  # Shape: (2^N, N+1)

    # Compute averages for each path
    averages = compute_averages(Sbar)  # Shape: (2^N,)

    # Compute payoffs for each path
    call_payoffs, put_payoffs = compute_payoffs(Sbar, averages)  # Shape: (2^N,)

    # Compute path probabilities
    path_probs = compute_path_probabilities(pu, N, paths)  # Shape: (2^N,)

    # Discount factor for the entire period
    discount_factor = R ** N

    # Compute expected option prices
    C0 = np.sum(call_payoffs * path_probs) / discount_factor
    P0 = np.sum(put_payoffs * path_probs) / discount_factor

    return C0, P0

# Testing the function
if __name__ == "__main__":
    C0, P0 = CRRfltAD(1, 90, 0.02, 0.20, 4)
    print(f"Floating Strike Call Option Price: {C0:.4f}")
    print(f"Floating Strike Put Option Price: {P0:.4f}")
