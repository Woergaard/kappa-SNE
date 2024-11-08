import numpy as np
from scipy.spatial.distance import cdist

def knn_error(train_X, train_labels, test_X, test_labels, k=1):
    # Compute pairwise distance matrix
    D = cdist(test_X, train_X, metric='sqeuclidean')

    # Perform labeling
    classification = np.zeros(test_X.shape[0], dtype=int)
    for j in range(test_X.shape[0]):
        ii = np.argsort(D[j])[:k]
        tmp1 = train_labels[ii]
        tmp2 = np.unique(tmp1)
        counts = np.array([np.sum(tmp1 == t) + np.sum(0.01 / np.where(tmp1 == t)[0]) for t in tmp2])
        classification[j] = tmp2[np.argmax(counts)]

    # Evaluate error
    err = np.sum(test_labels != classification) / len(test_labels)
    return err