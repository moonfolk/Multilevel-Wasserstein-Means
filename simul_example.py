import sys
import os

## Setting up paths
cur_dir = os.path.dirname(os.path.realpath('simul_example.py'))
path_cython = cur_dir + '/cython_cuturi'
sys.path.insert(0, cur_dir)
sys.path.insert(0, path_cython)

from W_means_class import W_means, simulate_mixture_nc, simulate_mixture_lc
import numpy as np

### Simulation experiments
np.random.seed(1)
d = 10 # dimensions of observations
M = 2000 # number of groups
#N = np.random.choice([50, 100, 150], M) # number of observations in groups
N = 50

### Set Multilevel K-means parameters
K = 5 # number of global barycenters
K_a = 6 # number of atoms in global barycenters (can be list)
k = 4 # number of atoms in local barycenters (can be list)
k_S = 50 # number of atoms in constraint set
n_iter = 10 # number of iterations to run
var = True # wheather to use Gaussian with non-constant variance

## Generating NC model. Z is the data, rest of the output is used to evaluate the fit
print '\nRunning NC simulations'
h_atoms, h_probs, labels_M, atoms_M, g_probs, Z = simulate_mixture_nc(K, K_a, d, N, M, k, var=var)
truth = [h_atoms, h_probs, atoms_M, g_probs]

# Fitting NC
print '\nFitting MWM'
nc_cluster = W_means(K=K, K_a=K_a, k=k, n_iter=n_iter, method='NC', verbose=1).fit(Z)
print 'NC Wasserstain distance to true model is %f' % -nc_cluster.score(truth)
nc_loc_labels = nc_cluster.loc_label(Z)

# Fitting LC
print '\nFitting MWMS'
lc_cluster = W_means(K=K, K_a=K_a, k=k_S, n_iter=n_iter, method='LC', verbose=1).fit(Z)
print 'LC Wasserstain distance to true model is %f' % -lc_cluster.score(truth)
lc_loc_labels = lc_cluster.loc_label(Z)

# Fitting multistage k-means
print '\nFitting 3-stage K-means'
k_cluster = W_means(K=K, K_a=K_a, k=k, n_iter=n_iter, method='3means', verbose=1).fit(Z)
print '3-means Wasserstain distance to true model is %f' % -k_cluster.score(truth)
k_loc_labels = k_cluster.loc_label(Z)

## Generating LC model. Z is the data, rest of the output is used to evaluate the fit
print '\nRunning LC simulations'
h_atoms, h_probs, labels_M, S_set, g_probs, Z = simulate_mixture_lc(K, K_a, d, N, M, k_S, var=var)
truth = [h_atoms, h_probs, M*[S_set], g_probs]

# Fitting NC
print '\nFitting MWM'
nc_cluster = W_means(K=K, K_a=K_a, k=k, n_iter=n_iter, method='NC', verbose=1).fit(Z)
print 'NC Wasserstain distance to true model is %f' % -nc_cluster.score(truth)

# Fitting LC
print '\nFitting MWMS'
lc_cluster = W_means(K=K, K_a=K_a, k=k_S, k_init=k, n_iter=n_iter, method='LC', verbose=1).fit(Z)
print 'LC Wasserstain distance to true model is %f' % -lc_cluster.score(truth)

# Fitting multistage k-means
print '\nFitting 3-stage K-means'
k_cluster = W_means(K=K, K_a=K_a, k=k, n_iter=n_iter, method='3means', verbose=1).fit(Z)
print '3-means Wasserstain distance to true model is %f' % -k_cluster.score(truth)
