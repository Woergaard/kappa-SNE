import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from multiprocessing import Pool, cpu_count
from kSNE import kSNE
import os

# Load datasets
digits = datasets.load_digits()
swiss_roll, swiss_roll_target = datasets.make_swiss_roll(n_samples=1000)
#mnist = datasets.fetch_openml('mnist_784')
#coil20 = datasets.fetch_olivetti_faces()  # Assuming COIL-20 is available as olivetti_faces
#olivetti_faces = datasets.fetch_olivetti_faces()
#fashion_mnist = datasets.fetch_openml('Fashion-MNIST')

datasets_dict = {
    'digits': {'data': digits.data, 'target': digits.target},
    'swiss_roll': {'data': swiss_roll, 'target': swiss_roll_target},
    #'mnist': {'data': mnist.data, 'target': mnist.target.astype(int)},
    #'coil20': {'data': coil20.data, 'target': coil20.target},
    #'olivetti_faces': {'data': olivetti_faces.data, 'target': olivetti_faces.target},
    #'fashion_mnist': {'data': fashion_mnist.data, 'target': fashion_mnist.target.astype(int)}
}

def process_task(args):
    n, kappa, beta, dataset_name = args
    data = datasets_dict[dataset_name]['data']
    target = datasets_dict[dataset_name]['target']

    try:
        ksne = kSNE(n_components=2, k=kappa, beta=beta, verbose=1, n_iter_without_progress=300, learning_rate=n)
        X_reduced = ksne.fit_transform(data)

        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=target)
        plt.title(f'$\kappa$ = {kappa:.3f}, lr = {n:.3f}, beta = {beta:.3f}')
        plt.savefig(f'fig/fig_{dataset_name}_{n:.3f}_{kappa:.3f}_{beta:.3f}.png')
        plt.close()  # Close the figure after saving to free up memory
    except Exception as e:
        with open('error_parameters.txt', 'a') as f:
            f.write(f"Error with parameters: n={n}, kappa={kappa}, beta={beta}, dataset_name={dataset_name}. Error message: {str(e)}\n")

if __name__ == '__main__':
    params = [(n, kappa, beta, dataset_name) 
          for n in np.arange(0.1, 2.1, 0.1) 
          for kappa in np.arange(0.1, 1.1, 0.1)
          for beta in np.arange(0.1, 2.1, 0.1) 
          for dataset_name in datasets_dict.keys()]

    # Remove configurations for which figures already exist
    params = [param for param in params if not os.path.exists(f'fig/fig_{param[3]}_{param[0]:.3f}_{param[1]:.3f}_{param[2]:.3f}.png')]

    # Use all available CPU cores
    with Pool(6) as pool:
        pool.map(process_task, params)


