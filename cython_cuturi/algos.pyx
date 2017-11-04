from __future__ import division
import numpy as np
cimport numpy as np
Df = np.float64
ctypedef np.float64_t Df_t
cimport cython
from libc.math cimport exp, log, sqrt, isinf, isnan
from libc.float cimport DBL_MAX, DBL_MIN



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.profile(False)
def algo3(np.ndarray[Df_t, ndim=1] a, np.ndarray[Df_t, ndim=1] b, np.ndarray[Df_t, ndim=2] M, double l = 10., str param='t', int max_iter=50, int verbose=0):
    cdef int n = M.shape[0]
    cdef int l_b = M.shape[1]
    cdef np.ndarray[Df_t, ndim=2] K = np.empty((n, l_b), dtype=Df)
    cdef np.ndarray[Df_t, ndim=2] K_til = np.empty((n, l_b), dtype=Df)
    cdef np.ndarray[Df_t, ndim=1] u = np.ones(n, dtype=Df)/n
    cdef int it = 0
    cdef int i, j
    cdef np.ndarray[Df_t, ndim=1] temp_v = np.empty(l_b, dtype=Df)
    cdef Df_t tmp
    cdef np.ndarray[Df_t, ndim=2] t = np.empty((n, l_b), dtype=Df)
    cdef np.ndarray[Df_t, ndim=1] alpha = np.empty(n, dtype=Df)
    cdef Df_t obj = 0
    
    for i in range(l_b):
        for j in range(n):
            tmp = exp(-l*M[j,i])
            K[j, i] = tmp
            tmp /= a[j]
            if isinf(tmp) or isnan(tmp):
                K_til[j, i] = DBL_MAX
            else:
                K_til[j, i] = tmp
    
    while it < max_iter:
        
        for i in range(l_b):
            tmp = 0
            for j in range(n):
                tmp += K[j,i]*u[j]
            tmp = b[i]/tmp
            if isinf(tmp):
                temp_v[i] = DBL_MAX
            elif isnan(tmp):
                temp_v[i] = 0
            else:    
                temp_v[i] = tmp # check for zero
        
        for j in range(n):
            tmp = 0
            for i in range(l_b):
                tmp += K_til[j,i] * temp_v[i]
            if tmp < DBL_MIN:
                u[j] = DBL_MAX
            else:
                u[j] = 1/tmp # check for zero

        it += 1
    if verbose:
        for j in range(n):
            tmp = 0
            for i in range(l_b):
                tmp += K[j,i] * M[j,i] * temp_v[i]
            obj += u[j]*tmp
    if param=='t':
        for i in range(l_b):
            for j in range(n):
                t[j,i] = K[j,i] * u[j] * temp_v[i]
        return t, obj
    elif param=='l':
        if not verbose:
            for j in range(n):
                tmp = 0
                for i in range(l_b):
                    tmp += K[j,i] * M[j,i] * temp_v[i]
                obj += u[j]*tmp
        return obj
    else:
        tmp = 0
        for j in range(n):
            if u[j]!=0:
                u[j] = log(u[j])
                tmp += u[j]
        tmp /= n*l
        for j in range(n):
            alpha[j] = u[j]/l - tmp
        return alpha, obj
        