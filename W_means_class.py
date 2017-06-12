from algos import algo3
from algos_cuturi import algo2
from algos_cuturi import algo1

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import time
import sklearn.metrics as metrics
from sklearn.base import BaseEstimator, ClusterMixin

######################## Update functions for different algorithms
## Updating atoms and weights of Gm
def y_update(z, w, Hi, ai, k, M, bm=None, Ym=None, weight=None, verbose=0, max_iter=[5, 10, 50]):
    Y = [z, Hi]
    b = [w, ai]
    return algo2(Y, b, n=k, a=bm, X=Ym, weight=weight, verbose=verbose, max_iter=max_iter)
    

## Assign group barycenters (Gm) to global barycenters    
def get_labels(Yi, bi, H, a, max_iter=50):
    K = len(H)
    M = [cdist(H[i].T, Yi.T, metric='euclidean') for i in range(K)]
    dist = [algo3(a[i], bi, M[i], param='l', max_iter=max_iter) for i in range(K)]
    return np.argmin(dist), min(dist)

## Compute distance to the true model
def objective_f(truth, model, max_iter=50):
    H_t, a_t, Y_t, b_t = truth
    H, a, Y, b = model
    M = len(Y)
    K_t = len(H_t)
    K = len(H)
    group_obj = sum([algo3(b_t[m], b[m], cdist(Y_t[m].T, Y[m].T, metric='euclidean'), param='l', max_iter=max_iter) for m in range(M)])/M
    H_to_t = [[algo3(a_t[i], a[k], cdist(H_t[i].T, H[k].T, metric='euclidean'), param='l', max_iter=max_iter) for i in range(K_t)] for k in range(K)]
    glob_obj = max([max(np.min(H_to_t, axis=0)), max(np.min(H_to_t, axis=1))])
    return group_obj + glob_obj

## Update weight for LC
def y_update_weight(z, w, Hi, ai, M, X, bm=None, weight=None, verbose=0, max_iter=[10,50]):
    Y = [z, Hi]
    b = [w, ai]
    M_dist = [cdist(X.T,Y[i].T, metric='euclidean') for i in range(len(Y))]
    return algo1(X, Y, b, M_dist, weight=weight, a_til=bm, verbose=verbose, max_iter=max_iter)
    

## Update set of atoms S in local constraint
def update_atoms(S, b, H, a, Z, W, labels, weight=[1.,1.], max_iter=50):
    M = len(Z)
    M_h = [cdist(H[labels[m]].T, S.T, metric='euclidean') for m in range(M)]
    M_z = [cdist(Z[m].T, S.T, metric='euclidean') for m in range(M)]
    T_h = [algo3(a[labels[m]], b[m], M_h[m], max_iter=max_iter)[0] for m in range(M)]
    T_z = [algo3(W[m], b[m], M_z[m], max_iter=max_iter)[0] for m in range(M)]
    k = S.shape[1]
    for l in range(k):
        z_part = weight[0]*np.sum([(z*t[:,l]).sum(axis=1) for (z,t) in zip(Z, T_z)], axis=0)
        z_weight = weight[0]*np.sum([t[:,l].sum() for t in T_z])
        h_part = weight[1]*np.sum([(h*th[:,l]).sum(axis=1) for (h,th) in zip([H[labels[m]] for m in range(M)], T_h)], axis=0)
        h_weight = weight[1]*np.sum([th[:,l].sum() for th in T_h])
        S[:,l] = (z_part + h_part)/(z_weight + h_weight)
  
#################### Learning functions for algorithms

## No constraint algorithm

## Initialization based on k-means
def init_nc(Z, K, K_a, k, n_iter=300, verbose=1):
    d = Z[0].shape[0] # dimension (assumed same everywhere)
    M = len(Z) # number of groups
    N = [Z[m].shape[1] for m in range(M)] # vector of group sizes
    
    if type(k) == int:
        k = np.repeat(k, M)
    if type(K_a) == int:
        K_a = np.repeat(K_a, K)
    
    ## Initialize with k-means++
    time_s = time.clock()
    kmeans = [KMeans(n_clusters=k[m], max_iter=n_iter).fit(Z[m].T) for m in range(M)]
    Y = [kmeans[m].cluster_centers_.T for m in range(M)]
    b = [np.unique(kmeans[m].labels_, return_counts=True)[1]*1./N[m] for m in range(M)]
    Y_flat = np.concatenate(Y, axis=1)
    kmeans = KMeans(n_clusters=K, max_iter=n_iter).fit(Y_flat.T)
    counts = np.unique(kmeans.labels_, return_counts=True)[1]
    H = []
    a = []
    for l in range(K):
        if counts[l] < K_a[l]:
            add_a = np.array([kmeans.cluster_centers_[l] + np.random.normal(size=d) for i in range(K_a[l]-counts[l])]).T
            H.append(np.append(Y_flat[:,kmeans.labels_==l], add_a, axis=1))
            a.append(np.repeat(1./K_a[l], K_a[l]))
        elif counts[l] == K_a[l]:
            H.append(Y_flat[:,kmeans.labels_==l])
            a.append(np.repeat(1./K_a[l], K_a[l]))
        else:
            k_means_init = KMeans(n_clusters=K_a[l], max_iter=n_iter).fit(Y_flat[:,kmeans.labels_==l].T)
            H.append(k_means_init.cluster_centers_.T)
            a.append(np.unique(k_means_init.labels_, return_counts=True)[1]*1./sum(kmeans.labels_==l))
    if verbose:
        print '3means initialization took %f' % (time.clock()-time_s)
    return Y, b, H, a

## Algorithm 1    
def learn_nc(Z, K, K_a, k, n_iter=20, weight=True, verbose=0, init = 'kmeans', n_iter_cuturi = [5, 10, 50]): # Data; #global means; atoms per global; atoms per local
    ## Data parameters
    d = Z[0].shape[0] # dimension (assumed same everywhere)
    M = len(Z) # number of groups
    N = [Z[m].shape[1] for m in range(M)] # vector of group sizes
    W = [np.repeat(1./N[m], N[m]) for m in range(M)] # weights of empirical destributions
    
    if type(k) == int:
        k = np.repeat(k, M)
    if type(K_a) == int:
        K_a = np.repeat(K_a, K)
    
    ## Improved initialization
    if init == 'kmeans':
        Y, b, H, a = init_nc(Z, K, K_a, k, verbose=verbose)
        print 'Done initializing with k-means'
    else:
        Y = [np.random.normal(size=(d,k[m])) for m in range(M)]
        b = [np.random.dirichlet(np.ones(k[m])) for m in range(M)]
        H = [np.random.normal(size=(d,K_a[i])) for i in range(K)]
        a = [np.random.dirichlet(np.ones(K_a[i])) for i in range(K)]
    
    ## Two options for importance weights for local updates - choose one
    if weight:
        weight_b = [1.*M/(M+1), 1./(M+1)]
    else:
        weight_b = None

    # Main part
    # Measure time
    time_l = 0.
    time_g = 0.
    time_h = 0.
    for it in range(n_iter):
        
        # Assign labels to each Gm
        time_s = time.clock()
        labels, obj = zip(*[get_labels(Y[m], b[m], H, a, max_iter=n_iter_cuturi[2]) for m in range(M)])
        time_e = time.clock()
        time_l += time_e - time_s
        if verbose:
            obj = np.sum(obj)
            print 'Sum of Gm to H objectives at iteration %d is %f' % (it, obj)
        
        # Update atoms and weights of each local barycenter (Gm)
        time_s = time.clock()
        Y, b, obj = zip(*[y_update(Z[m], W[m], H[labels[m]], a[labels[m]], k[m], M, bm=b[m], Ym=Y[m], weight = weight_b, verbose=verbose, max_iter=n_iter_cuturi) for m in range(M)])
        time_e = time.clock()
        time_g += time_e - time_s
        if verbose:
            obj = np.sum(obj)
            print 'Total objective is %f' % obj
        
        # Recompute labels
        time_s = time.clock()
        labels, obj = zip(*[get_labels(Y[m], b[m], H, a, max_iter=n_iter_cuturi[2]) for m in range(M)])
        time_e = time.clock()
        time_l += time_e - time_s
        
        # Update global barycenters (both atoms and weights)
        time_s = time.clock()
        for i in np.unique(labels):
            H[i], a[i], obj_i = algo2([Y[m] for m in range(M) if labels[m]==i], [b[m] for m in range(M) if labels[m]==i], n=K_a[i], X = H[i], a = a[i], max_iter=n_iter_cuturi)
        time_e = time.clock()
        time_h += time_e - time_s
        
    if verbose:
        print 'Updating labels took %f\nUpdating local barycenters took %f\nUpdating global barycenters took %f' % (time_l, time_g, time_h)
    return H, a, Y, b, labels # global atoms; global weights; local atoms; local weights; cluster assignments

## Local constraint algorithm

## Initialization based on k-means
def init_lc(Z, K, K_a, k, k_init, verbose=1):
    d = Z[0].shape[0] # dimension (assumed same everywhere)
    M = len(Z) # number of groups
    N = [Z[m].shape[1] for m in range(M)] # vector of group sizes
    Z_flat = np.concatenate(Z, axis=1)
    
    if type(K_a) == int:
        K_a = np.repeat(K_a, K)
    
    ## Initialize S with k-means++
    kmeans = KMeans(n_clusters=k).fit(Z_flat.T)
    S = kmeans.cluster_centers_.T
    b = []
    cur_ind = 0
    for m in range(M):
        l_m = kmeans.labels_[cur_ind:(cur_ind+N[m])]
        counts = np.unique(l_m, return_counts=True)
        cur_ind += N[m]
        b_0 = np.zeros(k)
        b_0[counts[0]] = 1.*counts[1]/N[m]
        b.append(b_0)
     
    ## Initialize H same as init_nc
    time_s = time.clock()
    kmeans = [KMeans(n_clusters=k_init).fit(Z[m].T) for m in range(M)]
    Y = [kmeans[m].cluster_centers_.T for m in range(M)]
    Y_flat = np.concatenate(Y, axis=1)
    kmeans = KMeans(n_clusters=K).fit(Y_flat.T)
    counts = np.unique(kmeans.labels_, return_counts=True)[1]
    H = []
    a = []
    for l in range(K):
        if counts[l] < K_a[l]:
            add_a = np.array([kmeans.cluster_centers_[l] + np.random.normal(size=d) for i in range(K_a[l]-counts[l])]).T
            H.append(np.append(Y_flat[:,kmeans.labels_==l], add_a, axis=1))
            a.append(np.repeat(1./K_a[l], K_a[l]))
        elif counts[l] == K_a[l]:
            H.append(Y_flat[:,kmeans.labels_==l])
            a.append(np.repeat(1./K_a[l], K_a[l]))
        else:
            k_means_init = KMeans(n_clusters=K_a[l]).fit(Y_flat[:,kmeans.labels_==l].T)
            H.append(k_means_init.cluster_centers_.T)
            a.append(np.unique(k_means_init.labels_, return_counts=True)[1]*1./sum(kmeans.labels_==l))
            
    if verbose:
        print '3means initialization took %f' % (time.clock()-time_s)
    return S, b, H, a

## Algorithm 2
def learn_lc(Z, K, K_a, k, n_iter=20, k_init = 5, weight = True, verbose=0, init = 'kmeans', n_iter_cuturi = [5, 10, 50]): # Data; #global means; atoms per global; atoms in constraint set 
    ## Data parameters
    d = Z[0].shape[0] # dimension (assumed same everywhere)
    M = len(Z) # number of groups
    N = [Z[m].shape[1] for m in range(M)] # vector of group sizes
    W = [np.repeat(1./N[m], N[m]) for m in range(M)] # weights of empirical destributions
    
    if type(K_a) == int:
        K_a = np.repeat(K_a, K)
        
    ## Improved initialization
    if init=='kmeans':
        S, b, H, a = init_lc(Z, K, K_a, k, k_init, verbose=verbose)
    else:
        S = np.random.normal(size=(d,k))
        b = [np.random.dirichlet(np.ones(k)) for m in range(M)]
        H = [np.random.normal(size=(d,K_a[i])) for i in range(K)]
        a = [np.random.dirichlet(np.ones(K_a[i])) for i in range(K)]

    # Weighting choice
    if weight:
        weight_a = [M,1.]
        weight_b = [1.*M/(M+1), 1./(M+1)]
    else:
        weight_a = [1.,1.]
        weight_b = None                   
         
    # Main part
    # Measure time
    time_l = 0.
    time_g = 0.
    time_h = 0.
    for it in range(n_iter):

        # Assign labels to each Gm
        time_s = time.clock()
        labels, obj = zip(*[get_labels(S, b[m], H, a, max_iter=n_iter_cuturi[2]) for m in range(M)])
        time_e = time.clock()
        time_l += time_e - time_s
        if verbose:
            obj = np.sum(obj)
            print 'Sum of Gm to H objectives at iteration %d is %f' % (it, obj)
        
        # Update set of atoms S and weights for each group's barycenter
        time_s = time.clock()
        update_atoms(S, b, H, a, Z, W, labels, weight=weight_a, max_iter=n_iter_cuturi[2])
        b, obj = zip(*[y_update_weight(Z[m], W[m], H[labels[m]], a[labels[m]], M, S, bm=b[m], weight=weight_b, verbose=verbose, max_iter=n_iter_cuturi[1:]) for m in range(M)])
        time_e = time.clock()
        time_g += time_e - time_s
        if verbose:
            obj = np.sum(obj)
            print 'Total objective is %f' % obj
        
        # Update global barycenters (both atoms and weights)
        time_s = time.clock()
        for i in np.unique(labels):
            H[i], a[i], obj_i = algo2([S for m in range(M) if labels[m]==i], [b[m] for m in range(M) if labels[m]==i], n=K_a[i], X=H[i], a=a[i], max_iter=n_iter_cuturi)
        time_e = time.clock()
        time_h += time_e - time_s
    
    if verbose:
        print 'Updating labels took %f\nUpdating local barycenters took %f\nUpdating global barycenters took %f' % (time_l, time_g, time_h)
    return H, a, S, b, labels 

####################### Data simulation functions

## Simulate from a mixture - utility
def sample_from_mixture(atoms, probs, size=1, var=1., dist='norm'):
    if probs.sum() != 1:
        probs = probs/probs.sum()
    atom_ind = np.random.choice(len(probs), size=size, p=probs)
    d = atoms.shape[0]
    if size>1:
        if dist=='norm':
            sample = np.random.normal(atoms[:,atom_ind], var)
        else:
            sample = np.random.exponential(atoms[:,atom_ind])
    else:
        if dist=='norm':
            sample = np.random.normal(atoms[:,atom_ind], var).reshape(d,)
        else:
            sample = np.random.exponential(atoms[:,atom_ind]).reshape(d,)
    return sample

## Simulate no constraint model
def simulate_mixture_nc(K, K_a, d, N, M, k, var=False, dist='norm'):
    if type(k) == int:
        k = np.repeat(k, M)
    if type(K_a) == int:
        K_a = np.repeat(K_a, K)
    if type(N) == int:
        N = np.repeat(N, M)
        
    means = np.arange(0., 1*K, 1)
    if dist != 'norm':
        means += 2
    
    if var:
        st_dev = np.sqrt(2*np.arange(1,K+1))
    else:
        st_dev = np.ones(K)
        
    if dist=='norm':
        global_atoms = [(means[i], 1., (d,K_a[i])) for i in range(K)]
        h_atoms = [np.random.normal(*global_atoms[i]) for i in range(K)]
    else:
        global_atoms = [(means[i], (d,K_a[i])) for i in range(K)]
        h_atoms = [np.random.exponential(*global_atoms[i]) for i in range(K)]
    h_probs = [np.random.dirichlet(np.ones(Ka)) for Ka in K_a]
    labels_M = np.random.choice(K, M)
    atoms_M = [sample_from_mixture(h_atoms[labels_M[m]], h_probs[labels_M[m]], k[m], st_dev[labels_M[m]], dist) for m in range(M)]
    g_probs = [np.random.dirichlet(np.ones(k[m])) for m in range(M)]
    Z = [sample_from_mixture(atoms_M[m], g_probs[m], N[m], st_dev[labels_M[m]], dist) for m in range(M)]
    return h_atoms, h_probs, labels_M, atoms_M, g_probs, Z

## Simulate local constraint model
def simulate_mixture_lc(K, K_a, d, N, M, k, var=False, dist='norm'):
    if type(K_a) == int:
        K_a = np.repeat(K_a, K)
    if type(N) == int:
        N = np.repeat(N, M)
        
    means = np.arange(0., 1*K, 1)
    if dist != 'norm':
        means += 2
    
    if var:
        st_dev = np.sqrt(2*np.arange(1,K+1))
    else:
        st_dev = np.ones(K)
        
    if dist=='norm':
        global_atoms = [(means[i], 1., (d,K_a[i])) for i in range(K)]
        h_atoms = [np.random.normal(*global_atoms[i]) for i in range(K)]
    else:
        global_atoms = [(means[i], (d,K_a[i])) for i in range(K)]
        h_atoms = [np.random.exponential(*global_atoms[i]) for i in range(K)]
    
    h_probs = [np.random.dirichlet(np.ones(Ka)) for Ka in K_a]
    labels_S = np.random.choice(K, k)
    S_set = np.array([sample_from_mixture(h_atoms[i], h_probs[i], var=st_dev[i], dist=dist) for i in labels_S]).T
    labels_M = np.random.choice(K, M)
    g_probs = []
    Z = []
    m = 0
    for l in labels_M:
        p = np.zeros(k)
        p[labels_S==l] = np.random.dirichlet(np.ones(sum(labels_S==l)))
        g_probs.append(p)
        data = sample_from_mixture(S_set, p, N[m], st_dev[l], dist)
        Z.append(data)
        m += 1
    return h_atoms, h_probs, labels_M, S_set, g_probs, Z

## Get local ind
def get_ind(atoms, weight, obs, thr=0.): # pass local atoms and observation
    dist = [((obs - atoms[:,k])**2).sum() if weight[k]>thr else np.inf for k in range(atoms.shape[1])]
    return np.argmin(dist)

## Algorithm class
class W_means(BaseEstimator, ClusterMixin):  
    def __init__(self, K=5, K_a=6, k=4, n_iter=10, weight=True, verbose=0, init = 'kmeans', k_init=5, method = 'NC', n_iter_cuturi = [5, 10, 50]):
        """
        Called when initializing the algorithm
        """
        self.K = K
        self.K_a = K_a
        self.k = k
        self.n_iter = n_iter
        self.weight = weight
        self.verbose = verbose
        self.init = init
        self.k_init = k_init
        self.method = method
        self.n_iter_cuturi = n_iter_cuturi

    def fit(self, Z, truth=None):
        
        self.truth = truth
        
        if self.method == 'NC':           
            self.H_, self.a_, self.Y_, self.b_, self.labels_ = \
                learn_nc(
                    Z, self.K, self.K_a, self.k, self.n_iter, self.weight, 
                    self.verbose, self.init, self.n_iter_cuturi)
        elif self.method == 'LC':
            self.H_, self.a_, self.S_, self.b_, self.labels_ = \
                learn_lc(
                    Z, self.K, self.K_a, self.k, self.n_iter, self.k_init, self.weight,
                    self.verbose, self.init, self.n_iter_cuturi)
            self.Y_ = len(Z)*[self.S_]
        else:
            self.Y_, self.b_, self.H_, self.a_ = init_nc(Z, self.K, self.K_a, self.k, n_iter=1000, verbose=self.verbose)
            self.labels_, _ = zip(*[get_labels(self.Y_[m], self.b_[m], self.H_, self.a_, max_iter=self.n_iter_cuturi[2]) for m in range(len(Z))])

        return self
    
    def fit_predict(self, Z):
        return self.fit(Z).labels_
              
    def loc_label(self, Z):
        return [[get_ind(y, b, x[:,i]) for i in range(x.shape[1])] for (x,b,y) in zip(*[Z, self.b_, self.Y_])]
    
    def score(self, truth=None):
        if self.truth == None:
            self.truth = truth

        if len(self.truth)==4:
            return -objective_f(self.truth, [self.H_, self.a_, self.Y_, self.b_])
        else:
            return metrics.adjusted_mutual_info_score(self.labels_,self.truth)