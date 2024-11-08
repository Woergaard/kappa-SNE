import numpy as np
from parametric.compute_recon_err import compute_recon_err

def train_rbm(X, h=30, eta=0.1, max_iter=30, weight_cost=0.0002):
    # Process inputs
    if isinstance(X, str):
        X = np.load(X)
    
    # Other parameters
    initial_momentum = 0.5
    final_momentum = 0.9
    
    # Initialize variables
    n, v = X.shape
    batch_size = 100
    machine = {
        'W': np.random.randn(v, h) * 0.1,
        'bias_upW': np.zeros((1, h)),
        'bias_downW': np.zeros((1, v))
    }
    deltaW = np.zeros((v, h))
    deltaBias_upW = np.zeros((1, h))
    deltaBias_downW = np.zeros((1, v))
    
    # Main loop
    ind = np.random.permutation(n)
    for iter in range(max_iter):
        # Set momentum
        momentum = initial_momentum if iter <= 5 else final_momentum
        
        # Run for all mini-batches
        for batch in range(0, n, batch_size):
            if batch + batch_size <= n:
                # Set values of visible nodes
                vis1 = X[ind[batch:min(batch + batch_size, n)], :]
                
                # Compute probabilities for hidden nodes
                hid1 = 1 / (1 + np.exp(-(vis1 @ machine['W'] + machine['bias_upW'])))
                
                # Sample states for hidden nodes
                hid_states = hid1 > np.random.rand(*hid1.shape)
                
                # Compute probabilities for visible nodes
                vis2 = 1 / (1 + np.exp(-(hid_states @ machine['W'].T + machine['bias_downW'])))
                
                # Compute probabilities for hidden nodes
                hid2 = 1 / (1 + np.exp(-(vis2 @ machine['W'] + machine['bias_upW'])))
                
                # Compute the weights update
                posprods = vis1.T @ hid1
                negprods = vis2.T @ hid2
                deltaW = momentum * deltaW + eta * (((posprods - negprods) / batch_size) - (weight_cost * machine['W']))
                deltaBias_upW = momentum * deltaBias_upW + (eta / batch_size) * (np.sum(hid1, axis=0) - np.sum(hid2, axis=0))
                deltaBias_downW = momentum * deltaBias_downW + (eta / batch_size) * (np.sum(vis1, axis=0) - np.sum(vis2, axis=0))
                
                # Update the network weights
                machine['W'] += deltaW
                machine['bias_upW'] += deltaBias_upW
                machine['bias_downW'] += deltaBias_downW
        
        # Estimate error
        siz = min(n, 5000)
        err = compute_recon_err(machine, X[:siz, :])
        print(f'Iteration {iter + 1} (rec. error = {err})...')
    
    print()
    
    # Compute error of trained RBM
    err = compute_recon_err(machine, X)
    
    return {
        'W': machine['W'],
        'bias_upW': machine['bias_upW'],
        'bias_downW': machine['bias_downW']
    }

    return machine, err