To run the Barnes-Hut implementation we make use of some of the scikit-learn functions which means that we need to run it in a virtual environment such that the correct packages are present. This can be done in the following way. 

```
python -m venv env
source env/bin/activate  # activate
pip install -U numpy scipy scikit-learn umap-learn cython keras pandas matplotlib tensorflow
```
To compile the cython code please use 
```
 python setup.py build_ext --inplace
```

To exit an environment write 
````
deactivate
````