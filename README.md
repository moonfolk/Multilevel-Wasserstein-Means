# Multilevel Clustering via Wasserstein Means

This is a Python 2 implementation of MWM and MWMS algorithms of Multilevel Clustering via Wasserstein Means (N. Ho, X. Nguyen, M. Yurochkin, H. Bui, V. Huynh, D. Phung); plus implementation of Algorithms 1, 2 and 3 of Fast Computation of Wasserstein Barycenters (M. Cuturi, A. Doucet). Code written by Mikhail Yurochkin.

## Overview

First compile Cython code in cython_cuturi folder. Install Anaconda and run (on Ubuntu):
```
cython algos.pyx
python setup.py build_ext --inplace
```

It implemets Algorithm 3 of Cuturi, which is the main computational routine.

algos_cuturi.py Implements Algorithm 1 and Algorithm 2 of Cuturi

W_means_class.py implements our clustering algoritms as a scikit-learn estimator

simul_example.py has some simulated examples

Implementation is designed to be used in the interactive mode (e.g. Python IDE like Spyder).

## Usage guide1

```
W_means(K=5, K_a=6, k=4, n_iter=10, weight=True, verbose=0, init = 'kmeans', k_init=5, method = 'NC', n_iter_cuturi = [5, 10, 50])
```

Parameters:

K: number of global clusters

K_a: number of atoms in global clusters (can be list of len(K_a) = K or integer)

k: number of atoms in local clusters (if method = 'LC' - number of atoms in the constraint set)

n_iter: number of iterations of the main algorithm

weight: Wheather to use a scaling factor 1/M when updating local barycenters. True is recommended

verbose: if 1, objective function value will be printed on every iteration and total running time

init: which initialization to use. 'kmeans' is recommended for 3stage kmeans initialization

k_init: used for 'kmeans' initialization of 'LC' method; Recomended - value of k if you were to fit 'NC' method

method: 'NC' for no constraint; 'LC' for local constraint; '3means' for 3 stage kmeans

n_iter_cuturi: Number of iterations to run for Cuturi [algo2; algo1; algo3]


Methods:
```
fit(data, truth=None)
```

data: list of length M (number of groups) of d x Nm arrays (d - dimensionality; Nm - number of points in group m)
truth: can be used by score function later

Returns:
H_: list of atoms of global barycenters  
a_: list of weights of atoms of global barycenters  
Y_: list of atoms of local barycenters (equal to len(Z)*[S] for LC method)  
b_: lsit of weights of atoms of local barycneters  
S_: array of atoms in the constraint set (only if method = 'LC')  
labels_: list of label assignments of groups to global clusters  

```
score(truth=None)
```

Note: can only be used on fitted object.  
Unless truth provided before, provide either true label assignments or true model parameters (later is used for simulations).

Returns:
Wasserstein distance to truth if true model parameters are given
AMI score if true label assignemtns are given
