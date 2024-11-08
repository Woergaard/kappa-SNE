import numpy as np

def trustworthiness(X, mappedX, k):
    n = X.shape[0]
    
    # Compute pairwise distance matrices
    sumX = np.sum(X ** 2, axis=1)
    hD = sumX[:, np.newaxis] + sumX - 2 * X @ X.T
    
    sumX = np.sum(mappedX ** 2, axis=1)
    lD = sumX[:, np.newaxis] + sumX - 2 * mappedX @ mappedX.T
    
    # Compute neighborhood indices
    ind1 = np.argsort(hD, axis=1)
    ind2 = np.argsort(lD, axis=1)
    
    # Compute trustworthiness values
    T = 0
    for i in range(n):
        ranks = np.array([np.where(ind1[i] == ind2[i, j + 1])[0][0] for j in range(k)])
        ranks = ranks - k - 1
        T += np.sum(ranks[ranks > 0])
    
    T = 1 - ((2 / (n * k * (2 * n - 3 * k - 1))) * T)
    return T