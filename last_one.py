import numpy as np

# CRR Parameters Function
def CRRparams(T, r, v, N):
    """Calculate CRR model parameters."""
    dt = T / N
    up = np.exp(v * np.sqrt(dt))
    down = 1 / up
    R = np.exp(r * dt)
    pu = (R - down) / (up - down)
    return pu, up, R

# Non-Recombining Tree for CRR Model
def NRTCRR(S0, up, down, N):
    """
    Generate the expanded stock price tree Sbar for the CRR model.
    """
    total_nodes = 2 ** (N + 1) - 1  # Total nodes in the tree
    Sbar = np.zeros(total_nodes)
    Sbar[0] = S0  # Root node

    for m in range((total_nodes - 1) // 2):
        left_child = 2 * m + 1
        right_child = 2 * m + 2
        Sbar[left_child] = Sbar[m] * down
        Sbar[right_child] = Sbar[m] * up

    return Sbar

# Function to Compute Running Minimum Along Paths
def NRTmin(Sbar):
    """
    Compute the running minimum stock price along each path.
    """
    total_nodes = len(Sbar)
    MinS = np.zeros_like(Sbar)
    MinS[0] = Sbar[0]

    for m in range((total_nodes - 1) // 2):
        left_child = 2 * m + 1
        right_child = 2 * m + 2
        MinS[left_child] = min(MinS[m], Sbar[left_child])
        MinS[right_child] = min(MinS[m], Sbar[right_child])

    return MinS

# CRR Ladder Put Option Pricing Function
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
    """
    # Compute CRR parameters
    pu, up, R = CRRparams(T, r, v, N)
    down = 1 / up

    # Generate the expanded stock price tree
    Sbar = NRTCRR(S0, up, down, N)
    MinS = NRTmin(Sbar)
    LadP = np.zeros_like(Sbar)

    # Initialize with the payoffs at expiry
    k = len(L)
    start = 2 ** N - 1
    end = 2 ** (N + 1) - 1

    for m in range(start, end):
        if MinS[m] > L[0]:
            LadP[m] = max(K - Sbar[m], 0)
        else:
            assigned = False
            for l in range(1, k):
                if MinS[m] > L[l]:
                    LadP[m] = max(max(K - Sbar[m], K - L[l - 1]), 0)
                    assigned = True
                    break  # Found the appropriate ladder level
            if not assigned:
                LadP[m] = max(max(K - Sbar[m], K - L[k - 1]), 0)

    # Backward recursion to compute option price at each node
    for m in range(start - 1, -1, -1):
        LadP[m] = (pu * LadP[2 * m + 2] + (1 - pu) * LadP[2 * m + 1]) / R

    # The option premium at t=0 is LadP[0]
    return LadP[0]

# CRR European Option Pricing Function
def CRReur(S0, K, T, r, sigma, N):
    """
    Computes the price of a European put option using the Cox-Ross-Rubinstein (CRR) model.
    
    Parameters:
    S0 (float): Initial spot price of the underlying asset.
    K (float): Strike price of the option.
    T (float): Time to expiry of the option in years.
    r (float): Risk-free interest rate (annualized).
    sigma (float): Volatility of the underlying asset (annualized).
    N (int): Number of time steps in the binomial model.
    
    Returns:
    float: The European put option price at time t=0.
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize asset prices at maturity
    j = np.arange(N + 1)
    asset_prices = S0 * u ** j * d ** (N - j)
    put_values = np.maximum(K - asset_prices, 0)
    
    # Backward induction to calculate option price at time t=0
    for i in range(N - 1, -1, -1):
        put_values = np.exp(-r * dt) * (p * put_values[1:i + 2] + (1 - p) * put_values[0:i + 1])
    
    return put_values[0]


# Main Execution for Comparison
if __name__ == "__main__":
    # Parameters
    T = 1          # Expiration time in years
    S0 = 50        # Spot stock price
    K = 55         # Strike price
    L = [45, 40, 35]  # Ladder levels (decreasing below S0 and K)
    r = 0.05       # Risk-free yield per year
    v = 0.20       # Volatility (>0)
    N = 4          # Number of steps in the tree

    # Calculate the ladder put option price
    ladder_put_price = CRRladP(T, S0, K, L, r, v, N)
    print(f"Ladder Put Option Price: {ladder_put_price:.4f}")

    # Calculate the European put option price
    european_put_price = CRReur(S0, K, T, r, v, N)
    print(f"European Put Option Price: {european_put_price:.4f}")
