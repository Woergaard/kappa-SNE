
# $\kappa$-SNE: Kaniadakis-Gaussian Distributed Stochastic Neighbor Embedding

$\kappa$-SNE is a tool designed for the visualization of high-dimensional data. It operates by converting similarities between data points into joint probabilities and aims to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data.

## Features

- Visualization of high-dimensional data in lower dimensions (usually 2D or 3D).
- Utilizes the Kaniadakis-Gaussian distribution for transforming the data.
- Allows customization through various parameters to suit different data types and preferences.

## Installation

Ensure that you have Python 3 and pip installed on your system. You can then install the required packages using the following command:

```bash
pip install -r requirements.txt
```
 
## Compiling Cython Code

To compile the Cython code, run the following command:

```bash
python3 setup.py build_ext --inplace
```

## Usage

Here is a simple example to get you started with using $\kappa$-SNE:

### Example

```python
import numpy as np
from kSNE import kSNE

# Sample high-dimensional data
X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

# Transforming the data using k-SNE
X_embedded = kSNE(n_components=2).fit_transform(X)

# Checking the shape of the transformed data
print(X_embedded.shape)  # Should print: (4, 2)
```

You can also try to run the `exampleCode.py` for an example of a visualization. 

## Parameters

- `n_components` (int, optional, default: 2): Dimension of the embedded space.
- `perplexity` (float, optional, default: 30): Related to the number of nearest neighbors, with typical values between 5 and 50.
- `k` (float, optional, default: 0.5): Hyperparameter of the Kaniadakis-Gaussian distribution, where 0 < k < 1.
- `beta` (float, optional, default: 1.0): Shape parameter for the Kaniadakis-Gaussian distribution, must be greater than 0.
- `early_exaggeration` (float, optional, default: 12.0): Controls cluster spacing in the embedded space.
- `learning_rate`: float or “auto”, default=”auto”:  The learning rate for optimization, typically in the range [10.0, 1000.0]. The ‘auto’ option sets the learning_rate to max(N / early_exaggeration / 4, 50), where N is the sample size.
- `n_iter` (int, optional, default: 1000): Maximum number of iterations for optimization.
- `n_iter_without_progress` (int, optional, default: 300): Maximum iterations without progress before aborting.
- `min_grad_norm` (float, optional, default: 1e-7): Minimum gradient norm required to continue optimization.
- `metric` (string or callable, optional): Metric for calculating distance between data points.
- `init` (string or numpy array, optional, default: "random"): Method for initialization of embedding.
- `verbose` (int, optional, default: 0): Verbosity level.
- `random_state` (int or RandomState instance, optional, default=None): Determines the random number generator.
- `n_jobs` (int or None, optional, default=None): The number of parallel jobs to run for neighbors search.

## Dependencies

- numpy
- scipy
- cython
- scikit-learn

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions or feedback, please open an issue in this repository, and we will get back to you as soon as possible.
