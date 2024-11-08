import numpy as np
import pandas as pd
from parametric.train_rbm import train_rbm
from parametric.train_rbm_pcd import train_rbm_pcd
from parametric.train_lin_rbm import train_lin_rbm
from parametric.ksne_v_backprop import ksne_v_backprop

def train_par_ksne(train_X, train_labels, test_X, test_labels, layers, training='CD1'):
    # Convert input data to numpy arrays if they're DataFrames
    if isinstance(train_X, pd.DataFrame):
        train_X = train_X.values
    if isinstance(test_X, pd.DataFrame):
        test_X = test_X.values

    # Ensure inputs are numpy arrays
    train_X = np.array(train_X)
    test_X = np.array(test_X)

    origX = train_X.copy()
    no_layers = len(layers)
    network = []  # Initialize network as an empty list

    for i in range(no_layers):
        print(f'Training layer {i + 1} (size {train_X.shape[1]} -> {layers[i]})...')

        if i != no_layers - 1:
            if training == 'CD1':
                network.append(train_rbm(train_X, layers[i]))
            elif training == 'PCD':
                network.append(train_rbm_pcd(train_X, layers[i]))
            elif training == 'None':
                v = train_X.shape[1]
                network.append({
                    'W': np.random.randn(v, layers[i]) * 0.1,
                    'bias_upW': np.zeros((1, layers[i])),
                    'bias_downW': np.zeros((1, v))
                })
            else:
                raise ValueError('Unknown training procedure.')

            train_X = 1 / (1 + np.exp(-(train_X @ network[i]['W'] + network[i]['bias_upW'])))
        else:
            if training != 'None':
                network.append(train_lin_rbm(train_X, layers[i]))
            else:
                v = train_X.shape[1]
                network.append({
                    'W': np.random.randn(v, layers[i]) * 0.1,
                    'bias_upW': np.zeros((1, layers[i])),
                    'bias_downW': np.zeros((1, v))
                })

    # Perform backpropagation of the network using k-sne gradient
    network, err = ksne_v_backprop(network[:no_layers], origX, train_labels, test_X, test_labels, 30, 30)

    return network, err