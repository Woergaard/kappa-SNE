import numpy as np

def compute_recon_err(machine, X):
    if isinstance(machine, list):
        # Run for every layer in the network
        err = np.zeros(len(machine))
        vis = X.astype(float)
        for i, layer in enumerate(machine):
            # Compute probabilities for hidden nodes
            if i < len(machine) - 1:
                hid = 1 / (1 + np.exp(-(vis @ layer['W'] + layer['bias_upW'])))
            else:
                hid = vis @ layer['W'] + layer['bias_upW']
            
            # Compute probabilities for visible nodes
            rec = 1 / (1 + np.exp(-(hid @ layer['W'].T + layer['bias_downW'])))
            
            # Compute reconstruction error
            err[i] = np.sum((vis - rec) ** 2) / X.shape[0]
            vis = hid
    else:
        # Compute probabilities for hidden nodes
        hid = 1 / (1 + np.exp(-(X.astype(float) @ machine['W'] + machine['bias_upW'])))
        
        # Compute probabilities for visible nodes
        rec = 1 / (1 + np.exp(-(hid @ machine['W'].T + machine['bias_downW'])))
        
        # Compute reconstruction error
        err = np.sum((X - rec) ** 2) / X.shape[0]
    
    return err