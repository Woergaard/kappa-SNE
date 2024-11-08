import numpy as np

def train_lin_rbm(X, h=20, eta=0.001, max_iter=50, weight_cost=0.0002):
    if isinstance(X, str):
        X = np.load(X)
    
    # Other parameters
    initial_momentum = 0.5
    final_momentum = 0.9
    
    # Initialize some variables
    n, v = X.shape
    batch_size = 100
    W = np.random.randn(v, h) * 0.1
    bias_upW = np.zeros((1, h))
    bias_downW = np.zeros((1, v))
    deltaW = np.zeros((v, h))
    deltaBias_upW = np.zeros((1, h))
    deltaBias_downW = np.zeros((1, v))
    
    # Main loop
    for iter in range(max_iter):
        # Set momentum
        err = 0
        ind = np.random.permutation(n)
        momentum = initial_momentum if iter <= 5 else final_momentum
        
        # Run for all mini-batches
        for batch in range(0, n, batch_size):
            if batch + batch_size <= n:
                # Set values of visible nodes
                vis1 = X[ind[batch:min(batch + batch_size, n)], :]
                
                # Compute probabilities for hidden nodes
                hid1 = vis1 @ W + bias_upW
                
                # Sample states for hidden nodes
                hid_states = hid1 + np.random.randn(*hid1.shape)
                
                # Compute probabilities for visible nodes
                vis2 = 1 / (1 + np.exp(-(hid_states @ W.T + bias_downW)))
                
                # Compute probabilities for hidden nodes
                hid2 = vis2 @ W + bias_upW
                
                # Now compute the weights update
                posprods = vis1.T @ hid1
                negprods = vis2.T @ hid2
                deltaW = momentum * deltaW + eta * (((posprods - negprods) / batch_size) - (weight_cost * W))
                deltaBias_upW = momentum * deltaBias_upW + (eta / batch_size) * (np.sum(hid1, axis=0) - np.sum(hid2, axis=0))
                deltaBias_downW = momentum * deltaBias_downW + (eta / batch_size) * (np.sum(vis1, axis=0) - np.sum(vis2, axis=0))
                
                # Update the network weights
                W += deltaW
                bias_upW += deltaBias_upW
                bias_downW += deltaBias_downW
                
                # Estimate the error
                err += np.sum((vis1 - vis2) ** 2)
        
        # Estimation of reconstruction error
        print(f'Iteration {iter + 1}: error ~{err / n}')
    
    # Return RBM
    machine = {
        'W': W,
        'bias_upW': bias_upW,
        'bias_downW': bias_downW
    }
    print()
    return machine