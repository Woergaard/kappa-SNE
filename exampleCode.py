import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from kSNE import kSNE
import os

# Load datasets
digits = datasets.load_digits()

datasets_dict = {
    'digits': {'data': digits.data, 'target': digits.target}
}

data = datasets_dict['digits']['data']
target = datasets_dict['digits']['target']

kappa = 0.7
beta = 1.0

ksne = kSNE(n_components=2, k=kappa, beta=beta, verbose=1, n_iter_without_progress=300, learning_rate='auto')
X_reduced = ksne.fit_transform(data)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=target)
plt.title(f'$\kappa$ = {kappa:.3f}, beta = {beta:.3f}')
plt.show() 
