import numpy as np
from itertools import product
from math import comb
import math 

def crr_params(T, r, v, N):
    dt = T / N
    up = np.exp(v * np.sqrt(dt))
    pu = (np.exp(r * dt) - 1 / up) / (up - 1 / up)
    R = np.exp(r * dt)
    return pu, up, R

def crr_eur_put(T, S0, K, r, v, N):
    pu, up, R = crr_params(T, r, v, N)
    dt = T / N
    # Generate stock price tree
    stock_prices = np.zeros((N+1, N+1))
    for i in range(N + 1):
        for j in range(i + 1):
            stock_prices[i, j] = S0 * (up ** (i - j)) * ((1 / up) ** j)
    
    # Initialize option values at maturity
    option_values = np.zeros((N+1, N+1))
    for j in range(N + 1):
        option_values[N, j] = max(K - stock_prices[N, j], 0)
    
    # Backward induction to calculate option value at each node
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[i, j] = (pu * option_values[i + 1, j] + 
                                   (1 - pu) * option_values[i + 1, j + 1]) / R
    return option_values

def crr_compound_put(T, T1, S0, K, L, r, v, N):
    pu, up, R = crr_params(T, r, v, N)
    P = crr_eur_put(T, S0, K, r, v, N)
    M = round(T1 * N / T)
    
    # Smaller output matrix for compound put option
    W = np.zeros((M+1, M+1))
    
    # Set terminal values at expiry T1
    for j in range(M + 1):
        W[M, j] = max(L - P[M, j], 0)
    
    # Backward induction for compound option pricing
    for n in range(M - 1, -1, -1):
        for j in range(n + 1):
            W[n, j] = (pu * W[n + 1, j] + (1 - pu) * W[n + 1, j + 1]) / R
    
    return W, P

def floating_strike_option_pricing(T, S0, r, v, N):
    pu, up, R = crr_params(T, r, v, N)
    dt = T / N

    # Build the stock price tree
    stock_tree = np.zeros((N+1, N+1))
    for i in range(N + 1):
        for j in range(i + 1):
            stock_tree[i, j] = S0 * (up ** (i - j)) * ((1 / up) ** j)
    
    # Initialize arrays for call and put values at maturity
    call_values_geom = np.zeros((N+1, N+1))
    put_values_geom = np.zeros((N+1, N+1))
    call_values_arith = np.zeros((N+1, N+1))
    put_values_arith = np.zeros((N+1, N+1))

    # Calculate option values at maturity using geometric and arithmetic means
    for j in range(N + 1):
        # Geometric mean of the path up to node (N, j)
        path_stock_prices = [stock_tree[i, min(i, j)] for i in range(j + 1)]
        geom_mean = np.exp(np.mean(np.log(path_stock_prices)))  # Geometric mean
        arith_mean = np.mean(path_stock_prices)                 # Arithmetic mean
        
        # Payoffs for floating strike options
        call_values_geom[N, j] = max(0, stock_tree[N, j] - geom_mean)
        put_values_geom[N, j] = max(0, geom_mean - stock_tree[N, j])
        call_values_arith[N, j] = max(0, stock_tree[N, j] - arith_mean)
        put_values_arith[N, j] = max(0, arith_mean - stock_tree[N, j])

    # Backward induction to calculate option prices at each node
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            call_values_geom[i, j] = (pu * call_values_geom[i + 1, j] + 
                                      (1 - pu) * call_values_geom[i + 1, j + 1]) / R
            put_values_geom[i, j] = (pu * put_values_geom[i + 1, j] + 
                                     (1 - pu) * put_values_geom[i + 1, j + 1]) / R
            call_values_arith[i, j] = (pu * call_values_arith[i + 1, j] + 
                                       (1 - pu) * call_values_arith[i + 1, j + 1]) / R
            put_values_arith[i, j] = (pu * put_values_arith[i + 1, j] + 
                                      (1 - pu) * put_values_arith[i + 1, j + 1]) / R

    # Return the floating strike option prices at the root node
    return {
        "Call_Geometric_Mean": call_values_geom[0, 0],
        "Put_Geometric_Mean": put_values_geom[0, 0],
        "Call_Arithmetic_Mean": call_values_arith[0, 0],
        "Put_Arithmetic_Mean": put_values_arith[0, 0]
    }
    
def CRRparams(T, r, v, N):
    """Calculate CRR parameters."""
    dt = T / N
    up = np.exp(v * np.sqrt(dt))
    pu = (np.exp(r * dt) - (1 / up)) / (up - (1 / up))
    R = np.exp(r * dt)
    return pu, up, R

def NRTCRR(S0, up, down, N):
    """Construct the stock price tree (Sbar) for CRR."""
    Sbar = np.zeros((2**(N + 1),))
    Sbar[1] = S0
    for i in range(1, N + 1):
        for j in range(2**i, 2**(i + 1)):
            if j % 2 == 0:
                Sbar[j] = Sbar[j // 2] * down
            else:
                Sbar[j] = Sbar[j // 2] * up
    return Sbar

def NRTpsums(log_Sbar, N):
    """Calculate partial sums of logs for the geometric mean."""
    lU = np.zeros((2**(N + 1),))
    for i in range(1, len(log_Sbar)):
        lU[i] = lU[i // 2] + log_Sbar[i]
    return lU

def CRRflg(T, S0, r, v, N):
    """Price floating strike options (Call and Put) using geometric means in CRR model."""
    pu, up, R = CRRparams(T, r, v, N)
    Sbar = NRTCRR(S0, up, 1 / up, N)
    lU = NRTpsums(np.log(Sbar), N)   # Partial sums of logs for geometric mean
    C = np.zeros(Sbar.shape)
    P = np.zeros(Sbar.shape)

    # Calculate payoffs at expiration
    for m in range(2**N, 2 * 2**N):
        Geom = np.exp(lU[m] / (N + 1))     # Geometric mean
        C[m] = max(0, Sbar[m] - Geom)      # Call payoff
        P[m] = max(0, Geom - Sbar[m])      # Put payoff

    # Price options by backward induction
    for m in range(2**N - 1, 0, -1):
        C[m] = (pu * C[2 * m + 1] + (1 - pu) * C[2 * m]) / R
        P[m] = (pu * P[2 * m + 1] + (1 - pu) * P[2 * m]) / R

    return C[1], P[1]  # Return the option premiums at t=0

def CRRflt(T, S0, r, v, N):
    """Price floating strike options (Call and Put) using arithmetic means in CRR model."""
    pu, up, R = CRRparams(T, r, v, N)
    Sbar = NRTCRR(S0, up, 1 / up, N)  # Stock price tree
    C = np.zeros(Sbar.shape)          # Initialize call option prices
    P = np.zeros(Sbar.shape)          # Initialize put option prices

    # Calculate payoffs at expiration with arithmetic mean
    for m in range(2**N, 2 * 2**N):
        # Compute the arithmetic mean of the path leading up to node `m`
        path_indices = []
        node = m
        for _ in range(N + 1):
            path_indices.append(node)
            node //= 2
        path_indices.reverse()
        Avg = np.mean(Sbar[path_indices])  # Arithmetic mean of the prices along the path
        C[m] = max(0, Sbar[m] - Avg)       # Call payoff
        P[m] = max(0, Avg - Sbar[m])       # Put payoff

    # Price options by backward induction
    for m in range(2**N - 1, 0, -1):
        C[m] = (pu * C[2 * m + 1] + (1 - pu) * C[2 * m]) / R
        P[m] = (pu * P[2 * m + 1] + (1 - pu) * P[2 * m]) / R

    return C[1], P[1]  # Return the option premiums at t=0

def CRRparams(T, r, v, N):
    """Calculate CRR parameters."""
    dt = T / N
    up = np.exp(v * np.sqrt(dt))
    pu = (np.exp(r * dt) - (1 / up)) / (up - (1 / up))
    R = np.exp(r * dt)
    return pu, up, R

def NRTCRR(S0, up, down, N):
    """Construct the stock price tree (Sbar) for CRR."""
    Sbar = np.zeros((2**(N + 1),))
    Sbar[1] = S0
    for i in range(1, N + 1):
        for j in range(2**i, 2**(i + 1)):
            if j % 2 == 0:
                Sbar[j] = Sbar[j // 2] * down
            else:
                Sbar[j] = Sbar[j // 2] * up
    return Sbar

def CRRaro(T, S0, K, r, v, N):
    """Compute average-rate Call and Put premiums in the CRR model."""
    pu, up, R = CRRparams(T, r, v, N)
    Sbar = NRTCRR(S0, up, 1 / up, N)  # Stock price tree
    C = np.zeros(Sbar.shape)          # Initialize call option prices
    P = np.zeros(Sbar.shape)          # Initialize put option prices

    # Calculate payoffs at expiration with arithmetic mean
    for m in range(2**N, 2 * 2**N):
        # Compute the arithmetic mean of the path leading up to node `m`
        path_indices = []
        node = m
        for _ in range(N + 1):
            path_indices.append(node)
            node //= 2
        path_indices.reverse()
        Avg = np.mean(Sbar[path_indices])  # Arithmetic mean of the prices along the path
        C[m] = max(0, Avg - K)             # Call payoff
        P[m] = max(0, K - Avg)             # Put payoff

    # Price options by backward induction
    for m in range(2**N - 1, 0, -1):
        C[m] = (pu * C[2 * m + 1] + (1 - pu) * C[2 * m]) / R
        P[m] = (pu * P[2 * m + 1] + (1 - pu) * P[2 * m]) / R

    return C[1], P[1]  # Return the option premiums at t=0

def CRRparams(T, r, v, N):
    """Calculate CRR parameters."""
    dt = T / N
    up = np.exp(v * np.sqrt(dt))
    pu = (np.exp(r * dt) - (1 / up)) / (up - (1 / up))
    R = np.exp(r * dt)
    return pu, up, R

def NRTCRR(S0, up, down, N):
    """Construct the stock price tree (Sbar) for CRR."""
    Sbar = np.zeros((2**(N + 1),))
    Sbar[1] = S0
    for i in range(1, N + 1):
        for j in range(2**i, 2**(i + 1)):
            if j % 2 == 0:
                Sbar[j] = Sbar[j // 2] * down
            else:
                Sbar[j] = Sbar[j // 2] * up
    return Sbar

def NRTpsums(Sbar, N):
    """Calculate partial sums for arithmetic mean."""
    Ubar = np.zeros(Sbar.shape)
    for i in range(1, len(Sbar)):
        Ubar[i] = Ubar[i // 2] + Sbar[i]
    return Ubar

def CRRflt(T, S0, r, v, N):
    """Price floating strike options (Call and Put) using arithmetic means in the CRR model."""
    pu, up, R = CRRparams(T, r, v, N)
    Sbar = NRTCRR(S0, up, 1 / up, N)  # Stock price tree
    C = np.zeros(Sbar.shape)          # Initialize call option prices
    P = np.zeros(Sbar.shape)          # Initialize put option prices

    # Calculate payoffs at expiration with arithmetic mean
    for m in range(2**N, 2 * 2**N):
        path_indices = []
        node = m
        for _ in range(N + 1):
            path_indices.append(node)
            node //= 2
        path_indices.reverse()
        Avg = np.mean(Sbar[path_indices])  # Arithmetic mean of the prices along the path
        C[m] = max(0, Sbar[m] - Avg)       # Call payoff
        P[m] = max(0, Avg - Sbar[m])       # Put payoff

    # Price options by backward induction
    for m in range(2**N - 1, 0, -1):
        C[m] = (pu * C[2 * m + 1] + (1 - pu) * C[2 * m]) / R
        P[m] = (pu * P[2 * m + 1] + (1 - pu) * P[2 * m]) / R

    return C[1], P[1]  # Return the option premiums at t=0

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

def NRTmax(Sbar, N):
    """
    Compute the maximum values along all paths in a non-recombining binary tree.

    Parameters:
    ----------
    Sbar : list or numpy.ndarray
        Array of stock prices arranged in a non-recombining binary tree.
        Length should be 2^N - 1.
    N : int
        Depth of the binary tree (must be >= 0).

    Returns:
    -------
    Maxb : numpy.ndarray
        Array containing the maximum stock price along each path.
        Same length as Sbar.
    """
    # Ensure Sbar is a NumPy array for efficient indexing
    Sbar = np.array(Sbar)
    
    # Total number of nodes in the binary tree
    total_nodes = 2**N - 1
    
    if len(Sbar) != total_nodes:
        raise ValueError(f"Sbar must have exactly {total_nodes} elements for N={N}.")
    
    # Initialize Maxb with zeros, same shape as Sbar
    Maxb = np.zeros_like(Sbar, dtype=float)
    
    # Set the root node's maximum to its own value
    Maxb[0] = Sbar[0]
    
    # Iterate through all parent nodes (indices 0 to 2^(N-1) -1)
    # These are the nodes that have children
    num_parent_nodes = (2**N - 1) // 2
    for p in range(num_parent_nodes):
        # Calculate children indices in zero-based Python indexing
        left_child = 2 * p + 1
        right_child = 2 * p + 2
        
        # Update Maxb for the left child
        if left_child < total_nodes:
            Maxb[left_child] = max(Maxb[p], Sbar[left_child])
        
        # Update Maxb for the right child
        if right_child < total_nodes:
            Maxb[right_child] = max(Maxb[p], Sbar[right_child])
    
    return Maxb

from math import comb

def CRRparams(T, r, v, N):
    """
    Calculate CRR model parameters: risk-neutral probability, up factor, and risk-free rate per step.

    Parameters:
    T (float): Expiration time in years.
    r (float): Risk-free interest rate per year.
    v (float): Volatility of the underlying asset.
    N (int): Number of steps in the binomial tree.

    Returns:
    tuple: (pu, u, R) where
        pu (float): Risk-neutral probability of an up move.
        u (float): Up factor.
        R (float): Risk-free rate per step.
    """
    dt = T / N
    u = math.exp(v * math.sqrt(dt))
    d = 1 / u
    R = math.exp(r * dt)
    pu = (R - d) / (u - d)
    return pu, u, R

def NRTCRR(S0, up, down, N):
    """
    Generate the expanded stock price tree Sbar for the CRR model.

    Returns:
        Sbar: list of lists, where each inner list contains the stock prices along a path.
    """
    paths = list(product(['u', 'd'], repeat=N))
    Sbar = []
    for path in paths:
        S = S0
        S_path = [S0]
        for move in path:
            if move == 'u':
                S *= up
            else:
                S *= down
            S_path.append(S)
        Sbar.append(S_path)
    return Sbar

def NRTmin(Sbar):
    """
    Compute the running minimum stock price along each path.

    Parameters:
        Sbar: list of lists, stock prices along each path.

    Returns:
        MinS: list of minimum stock prices along each path.
    """
    MinS = []
    for S_path in Sbar:
        MinS.append(min(S_path))
    return MinS

def NRTmax(Sbar):
    """
    Compute the running maximum stock price along each path.

    Parameters:
        Sbar: list of lists, stock prices along each path.

    Returns:
        MaxS: list of maximum stock prices along each path.
    """
    MaxS = []
    for S_path in Sbar:
        MaxS.append(max(S_path))
    return MaxS

def PathAD(pu, R, N):
    """
    Compute the Arrow-Debreu prices Lbar for each path.

    Returns:
        Lbar: list of Arrow-Debreu prices for each path.
    """
    pd = 1 - pu
    discount_factor = 1 / (R ** N)
    paths = list(product(['u', 'd'], repeat=N))
    Lbar = []
    for path in paths:
        k = path.count('u')
        # Since we are iterating over all possible paths, we do not multiply by comb(N, k)
        path_prob = (pu ** k) * (pd ** (N - k))
        L = path_prob * discount_factor
        Lbar.append(L)
    return Lbar

def CRRlbAD(T, S0, r, v, N):
    """
    Price Lookback Call and Put options using path-dependent Arrow-Debreu
    expansions with the Cox-Ross-Rubinstein (CRR) binomial pricing model.

    INPUTS:
    T  = expiration time in years
    S0 = spot stock price
    r  = riskless yield per year
    v  = volatility; must be >0
    N  = height of the tree

    OUTPUTS:
    C0 = Call option premium at t=0
    P0 = Put option premium at t=0
    """
    # Compute CRR parameters
    pu, up, R = crr_params(T, r, v, N)
    down = 1 / up
    pd = 1 - pu

    # Generate the expanded stock price tree Sbar
    Sbar = NRTCRR(S0, up, down, N)  # List of paths
    # Compute MinS and MaxS along each path
    MinS = NRTmin(Sbar)
    MaxS = NRTmax(Sbar)

    # Compute the Arrow-Debreu prices Lbar
    Lbar = PathAD(pu, R, N)

    # Extract the stock prices at expiry (final price along each path)
    Sbar_at_expiry = [S_path[-1] for S_path in Sbar]

    # Convert lists to numpy arrays
    Sbar_at_expiry = np.array(Sbar_at_expiry)
    MinS = np.array(MinS)
    MaxS = np.array(MaxS)
    Lbar = np.array(Lbar)

    # Compute option prices
    C0 = np.dot(Sbar_at_expiry - MinS, Lbar)  # Lookback Call option price
    P0 = np.dot(MaxS - Sbar_at_expiry, Lbar)  # Lookback Put option price

    return C0, P0

def CRRlb(T, S0, r, v, N):
    # Compute CRR parameters
    pu, up, R = CRRparams(T, r, v, N)
    down = 1 / up

    # Generate all possible paths
    paths = list(product([down, up], repeat=N))
    num_paths = len(paths)

    # Initialize arrays
    S_T = np.zeros(num_paths)
    MinS = np.zeros(num_paths)
    MaxS = np.zeros(num_paths)
    path_probs = np.zeros(num_paths)

    for idx, path in enumerate(paths):
        S_path = [S0]
        for move in path:
            S_path.append(S_path[-1] * move)
        S_T[idx] = S_path[-1]
        MinS[idx] = min(S_path)
        MaxS[idx] = max(S_path)

        num_up = sum(1 for move in path if move == up)
        num_down = N - num_up
        path_probs[idx] = (pu ** num_up) * ((1 - pu) ** num_down)

    # Discount factor
    discount_factor = np.exp(-r * T)

    # Compute option prices
    C0 = discount_factor * np.dot(path_probs, S_T - MinS)
    P0 = discount_factor * np.dot(path_probs, MaxS - S_T)

    return C0, P0

def CRRladP(T, S0, K, L, r, v, N):
    """
    Price a ladder put option using the Cox-Ross-Rubinstein (CRR) binomial model.

    Parameters:
    T (float): Expiration time in years.
    S0 (float): Spot stock price.
    K (float): Strike price.
    L (list or numpy.ndarray): Decreasing ladder levels below S0 and K.
    r (float): Risk-free yield per year.
    v (float): Volatility (>0).
    N (int): Number of steps in the tree.

    Returns:
    float: The ladder put option premium at time t=0.

    Example:
    LadP = CRRladP(1, 50, 55, [45, 40, 35], 0.05, 0.20, 4)
    """
    # Step 1: Compute CRR parameters
    pu, up, R = CRRparams(T, r, v, N)
    down = 1 / up

    # Step 2: Generate the expanded stock price tree
    Sbar = NRTCRR(S0, up, down, N)
    MinS = NRTmin(Sbar)
    LadP = np.zeros_like(Sbar)

    # Step 3: Initialize with the payoffs at expiry
    k = len(L)
    start = 2 ** N - 1
    end = 2 ** (N + 1) - 1

    for m in range(start, end):
        if MinS[m] > L[0]:
            LadP[m] = max(K - Sbar[m], 0)
        else:
            if MinS[m] > L[k - 1]:
                for l in range(1, k):
                    if MinS[m] > L[l]:
                        LadP[m] = max(max(K - Sbar[m], K - L[l - 1]), 0)
                        break  # Found the appropriate ladder level
            else:
                LadP[m] = max(max(K - Sbar[m], K - L[k - 1]), 0)

    # Step 4: Backward recursion to compute option price at each node
    for m in range(start - 1, -1, -1):
        LadP[m] = (pu * LadP[2 * m + 2] + (1 - pu) * LadP[2 * m + 1]) / R

    # The option premium at t=0 is LadP[0]
    return LadP[0]

def CRReur(S0, K, T, r, sigma, N):
    """
    Computes the prices of European-style call and put options using 
    the Cox-Ross-Rubinstein (CRR) model with backward induction.

    Parameters:
    S0 (float): Initial spot price of the underlying asset.
    K (float): Strike price of the options.
    T (float): Time to expiry of the options in years.
    r (float): Risk-free interest rate (annualized).
    sigma (float): Volatility of the underlying asset (annualized).
    N (int): Number of time steps in the binomial model.

    Returns:
    tuple: A tuple containing the call price and put price at time t=0.
    """
    
    # Calculate parameters for the binomial tree
    dt = T / N  # Time step size
    u = np.exp(sigma * np.sqrt(dt))  # Up factor for price increase
    d = 1 / u  # Down factor for price decrease
    p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability

    # Initialize asset prices and option values
    asset_prices = np.zeros((N + 1, N + 1))  # Asset prices at each node
    call_values = np.zeros((N + 1, N + 1))   # Call option values at each node
    put_values = np.zeros((N + 1, N + 1))    # Put option values at each node

    # Fill in the asset prices at maturity
    for i in range(N + 1):
        asset_prices[i, N] = S0 * (u ** (N - i)) * (d ** i)

    # Calculate option values at maturity
    for i in range(N + 1):
        call_values[i, N] = max(0, asset_prices[i, N] - K)  # Call option payoff
        put_values[i, N] = max(0, K - asset_prices[i, N])   # Put option payoff

    # Backward induction for call option values
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            call_values[i, j] = np.exp(-r * dt) * (p * call_values[i, j + 1] + (1 - p) * call_values[i + 1, j + 1])

    # Backward induction for put option values
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            put_values[i, j] = np.exp(-r * dt) * (p * put_values[i, j + 1] + (1 - p) * put_values[i + 1, j + 1])

    # The option price at time t=0 is found at the top of the trees
    call_price = call_values[0, 0]
    put_price = put_values[0, 0]

    return call_price, put_price  # Return the call and put prices at time t=0
