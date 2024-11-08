import numpy as np
from scipy.optimize import minimize
from parametric.x2p import x2p
from parametric.run_data_through_network import run_data_through_network
from parametric.knn_error import knn_error
from parametric.ksne_grad import ksne_grad

def ksne_backprop(network, train_X, train_labels, test_X, test_labels, max_iter=30, perplexity=30, v=None):
    if v is None:
        v = len(network[-1]['bias_upW']) - 1
    
    # Initialize some variables
    n = train_X.shape[0]
    batch_size = min(5000, n)
    ind = np.random.permutation(n)
    err = np.zeros(max_iter)
    
    # Precompute joint probabilities for all batches
    print('Precomputing P-values...')
    curX = []
    P = []
    for batch in range(0, n, batch_size):
        if batch + batch_size <= n:
            cur_batch = train_X[ind[batch:min(batch + batch_size, n)], :]
            curX.append(cur_batch)
            p = x2p(cur_batch, perplexity, 1e-5)
            p[np.isnan(p)] = 0
            p = (p + p.T) / 2
            p /= np.sum(p)
            p = np.maximum(p, np.finfo(float).eps)
            P.append(p)
    
    # Run the optimization
    for iter in range(max_iter):
        print(f'Iteration {iter + 1}...')
        b = 0
        for batch in range(0, n, batch_size):
            if batch + batch_size <= n:
                # Construct current solution
                x = []
                for layer in network:
                    x.extend(layer['W'].ravel())
                    x.extend(layer['bias_upW'].ravel())
                x = np.array(x)
                
                # Perform conjugate gradient using three line searches
                res = minimize(ksne_grad, x, args=(curX[b], P[b], network, v), method='CG', jac=True, options={'maxiter': 3})
                x = res.x
                b += 1
                
                # Store new solution
                ii = 0
                for i in range(len(network)):
                    w_shape = network[i]['W'].shape
                    b_shape = network[i]['bias_upW'].shape
                    network[i]['W'] = x[ii:ii + np.prod(w_shape)].reshape(w_shape)
                    ii += np.prod(w_shape)
                    network[i]['bias_upW'] = x[ii:ii + np.prod(b_shape)].reshape(b_shape)
                    ii += np.prod(b_shape)
        
        # Estimate the current error
        activations = run_data_through_network(network, curX[0])
        sum_act = np.sum(activations ** 2, axis=1)
        Q = (1 + (sum_act[:, np.newaxis] + sum_act - 2 * activations @ activations.T) / v) ** -((v + 1) / 2)
        np.fill_diagonal(Q, 0)
        Q /= np.sum(Q)
        Q = np.maximum(Q, np.finfo(float).eps)
        C = np.sum(P[0] * np.log(np.maximum(P[0], np.finfo(float).eps) / np.maximum(Q, np.finfo(float).eps)))
        print(f'k-sne error: {C}')
        
        # Compute current 1-NN error
        err[iter] = knn_error(run_data_through_network(network, train_X), train_labels,
                              run_data_through_network(network, test_X), test_labels, 1)
        print(f'1-NN error: {err[iter]}')
    
    return network, err