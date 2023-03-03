import numpy as np
from scipy.spatial.distance import cdist

from algos import algo3

## Algorithm 1 of Cuturi - computes weights for barycenter with fixed atoms
#Input X - atoms of barycenter, Y - list of atoms of distributions, b - lsit of weights of distributions, M - distance matrix, l - see Cuturi paper
def algo1(X, Y, b, M, weight=None, a_til=None, verbose=0, max_iter=[10,50]):
    d, n = X.shape
    N = len(Y)
    
    # Initializing importance weights and weights of barycenter unless provided
    if weight is None:
        weight = np.repeat(1./N, N)
    if a_til is None or sum(a_til==1)>0: # second part is to avoid being stuck at extreme case
        a_til = np.ones((n,))/n
        
    a_hat = a_til
    t = 1.
    t_0 = 1.
    
    # Running optimization
    while t < max_iter[0]:
        beta = (t+1)/2
        a = (1-1/beta)*a_hat + a_til/beta
        algo_out = [algo3(a, b[i], M[i], param='alpha', verbose=verbose, max_iter=max_iter[1]) for i in range(N)]
        alpha = [weight[i]*algo_out[i][0] for i in range(N)]        
        alpha = np.sum(alpha, axis=0)
        a_til_n = a_til * np.exp(-t_0*beta*alpha)
        
        # Solving potential numeric issues
        if np.sum(np.isinf(a_til_n)) == 1:
            a_til = np.zeros((n,))
            a_til[np.isinf(a_til_n)] = 1.
        elif np.all(a_til_n==0):
            a_til = np.ones((n,))/n
        else:
            a_til = a_til_n/a_til_n.sum()
            
        a_hat = (1-1/beta)*a_hat + a_til/beta
        if np.any(np.isnan(a_hat)):
            print('Something is wrong in Algo1 Cuturi') 
        t += 1
    obj = 0
    if verbose:
        obj = np.sum([algo_out[i][1] for i in range(N)])
    return a_hat, obj
     
## Algorithm 2 of Cuturi - computes barycenter
#Input Y - list of atoms of distribtuions, b - lsit of weights of distributions, n - number of atoms in barycenter
def algo2(Y, b, n, weight=None, X=None, a=None, verbose=0, max_iter=[5, 10, 50]):
    N = len(Y)
    d = Y[0].shape[0]
    
    # Initializing importance weights, atoms of barycenter and weights of barycenter unless provided
    if weight is None:
        weight = np.repeat(1./N, N)
    if X is None:
        X = np.random.normal(3, 5, (d,n))
    if a is None:
        a = np.random.dirichlet(np.ones(n)*1.)

    c = 1
    
    # Running optimization
    while c < max_iter[0]:
        teta = 1./c
        M = [cdist(X.T,Y[i].T, metric='euclidean') for i in range(N)]
        a, _ = algo1(X, Y, b, M, weight=weight, a_til=a, verbose=verbose, max_iter=max_iter[1:])
        algo_out = [algo3(a, b[i], M[i], verbose=verbose, max_iter=max_iter[2]) for i in range(N)]
        t = [weight[i]*np.dot(Y[i],algo_out[i][0].T) for i in range(N)]
        t = np.sum(t, axis=0)/a[None,:]
        X = (1-teta)*X + teta*t
        c+=1
        if np.any(np.isnan(X)):
            print('Something is wrong in Algo2 Cuturi') 
    obj = 0
    if verbose:
        obj = np.sum([algo_out[i][1] for i in range(N)])
    return X, a, obj