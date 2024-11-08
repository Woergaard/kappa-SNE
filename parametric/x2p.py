import numpy as np

def x2p(X, u=15, tol=1e-4):
    n = X.shape[0]
    P = np.zeros((n, n))
    beta = np.ones(n)
    logU = np.log(u)
    
    # Compute pairwise distances
    print('Computing pairwise distances...')
    sum_X = np.sum(X ** 2, axis=1)
    D = sum_X[:, np.newaxis] + sum_X - 2 * X @ X.T
    
    # Run over all datapoints
    print('Computing P-values...')
    for i in range(n):
        if i % 500 == 0:
            print(f'Computed P-values {i} of {n} datapoints...')
        
        # Set minimum and maximum values for precision
        betamin = -np.inf
        betamax = np.inf
        
        # Compute the Gaussian kernel and entropy for the current precision
        Di = D[i, np.arange(n) != i]
        H, thisP = Hbeta(Di, beta[i])
        
        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i]
                if np.isinf(betamax):
                    beta[i] *= 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i]
                if np.isinf(betamin):
                    beta[i] /= 2
                else:
                    beta[i] = (beta[i] + betamin) / 2
            
            # Recompute the values
            H, thisP = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1
        
        # Set the final row of P
        P[i, np.arange(n) != i] = thisP
    
    print(f'Mean value of sigma: {np.mean(np.sqrt(1 / beta))}')
    print(f'Minimum value of sigma: {np.min(np.sqrt(1 / beta))}')
    print(f'Maximum value of sigma: {np.max(np.sqrt(1 / beta))}')
    
    return P, beta

def Hbeta(D, beta):
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P /= sumP
    return H, P