import numpy as np

def ksne_v_grad(x, X, P, network):
    # Initialize some variables
    n = X.shape[0]
    no_layers = len(network)

    # Update the network to store the new solution
    v = x[0]
    ii = 1
    for i in range(len(network)):
        w_shape = network[i]['W'].shape
        b_shape = network[i]['bias_upW'].shape
        network[i]['W'] = x[ii:ii + np.prod(w_shape)].reshape(w_shape)
        ii += np.prod(w_shape)
        network[i]['bias_upW'] = x[ii:ii + np.prod(b_shape)].reshape(b_shape)
        ii += np.prod(b_shape)

    # Run the data through the network
    activations = [np.column_stack((X, np.ones(n)))]
    for i in range(no_layers - 1):
        act = 1 / (1 + np.exp(-(activations[i] @ np.vstack((network[i]['W'], network[i]['bias_upW'])))))
        activations.append(np.column_stack((act, np.ones(n))))
    activations.append(activations[-1] @ np.vstack((network[-1]['W'], network[-1]['bias_upW'])))

    # Compute the Q-values
    sum_act = np.sum(activations[-1] ** 2, axis=1)
    D = sum_act[:, np.newaxis] + sum_act - 2 * activations[-1] @ activations[-1].T
    num = 1 + (D / v)
    Q = num ** -((v + 1) / 2)
    np.fill_diagonal(Q, 0)
    Q /= np.sum(Q)
    Q = np.maximum(Q, np.finfo(float).eps)
    num = 1 / num
    np.fill_diagonal(num, 0)

    # Compute the value of the cost function
    C = np.sum(P * np.log(np.maximum(P, np.finfo(float).eps) / np.maximum(Q, np.finfo(float).eps)))

    # Compute the derivatives w.r.t. the map coordinates (= errors)
    Ix = np.zeros_like(activations[-1])
    stiffnesses = (4 * ((v + 1) / (2 * v))) * (P - Q) * num
    for i in range(n):
        Ix[i, :] = np.sum(stiffnesses[:, i][:, np.newaxis] * (activations[-1][i] - activations[-1]), axis=0)

    # Compute gradients w.r.t. weights
    dW = [None] * no_layers
    db = [None] * no_layers
    for i in range(no_layers - 1, -1, -1):
        delta = activations[i].T @ Ix
        dW[i] = delta[:-1, :]
        db[i] = delta[-1, :]

        if i > 0:
            Ix = (Ix @ np.vstack((network[i]['W'], network[i]['bias_upW'])).T) * activations[i] * (1 - activations[i])
            Ix = Ix[:, :-1]

    # Compute gradient w.r.t. degrees of freedom
    dV = -np.sum((P - Q) * -((-v - 1) * D) / (2 * v**2 * (1 + D / v)) + 0.5 * np.log(1 + D / v))

    # Convert gradient information
    dC = np.zeros(len(x))
    dC[0] = dV
    ii = 1
    for i in range(no_layers):
        dC[ii:ii + np.prod(dW[i].shape)] = dW[i].ravel()
        ii += np.prod(dW[i].shape)
        dC[ii:ii + np.prod(db[i].shape)] = db[i].ravel()
        ii += np.prod(db[i].shape)

    return C, dC