import numpy as np

def run_data_through_network(network, X):
    if isinstance(network, dict):
        network = [network]
    
    # Run the data through the network
    n = X.shape[0]
    mappedX = np.column_stack((X, np.ones(n)))
    for i in range(len(network) - 1):
        mappedX = np.column_stack((1 / (1 + np.exp(-(mappedX @ np.vstack((network[i]['W'], network[i]['bias_upW']))))), np.ones(n)))
    mappedX = mappedX @ np.vstack((network[-1]['W'], network[-1]['bias_upW']))
    
    return mappedX