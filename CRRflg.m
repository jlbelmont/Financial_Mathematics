function [C, P] = CRRflg(T, S0, r, v, N)
    % Octave function to price floating strike Call and Put options with geometric means
    % using the Cox-Ross-Rubinstein (CRR) binomial model.
    
    % Set up parameters
    [pu, up, R] = CRRparams(T, r, v, N);
    Sbar = NRTCRR(S0, up, 1 / up, N); % expanded S tree
    lU = NRTpsums(log(Sbar), N); % partial sums of logs
    
    % Initialize option matrices
    C = zeros(size(Sbar));
    P = zeros(size(Sbar));
    
    % Calculate geometric mean and option payoff at expiry
    for m = 2^N : (2 * 2^N - 1)
        Geom = exp(lU(m) / (N + 1)); % geometric mean
        C(m) = max(0, Sbar(m) - Geom); % Call payoff
        P(m) = max(0, Geom - Sbar(m)); % Put payoff
    end
    
    % Backward induction
    for m = (2^N - 1) : -1 : 1
        C(m) = (pu * C(2 * m + 1) + (1 - pu) * C(2 * m)) / R;
        P(m) = (pu * P(2 * m + 1) + (1 - pu) * P(2 * m)) / R;
    end
end
