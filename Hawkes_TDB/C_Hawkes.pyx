import numpy as np
cimport numpy as np

def SUM_exp(double alpha, double beta, np.ndarray[np.float64_t,ndim=1] T, int n):
    
    cdef np.ndarray[np.float64_t,ndim=1] l    = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=1] dl_a = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=1] dl_b = np.zeros(n, dtype=np.float64)
    
    cdef np.ndarray[np.float64_t,ndim=1] dT = T[1:]-T[:-1]
    cdef np.ndarray[np.float64_t,ndim=1] r = np.exp(-beta*dT)
    
    cdef double x = 0.0
    cdef double x_a = 0.0
    cdef double x_b = 0.0
    
    for i in xrange(n-1):
        x   = ( x   + alpha*beta  ) * r[i] 
        x_a = ( x_a +       beta  ) * r[i]
        x_b = ( x_b + alpha       ) * r[i] - x*dT[i]
        
        l[   i+1] = x
        dl_a[i+1] = x_a
        dl_b[i+1] = x_b

    return [l,dl_a,dl_b]


def SUM_exp_L(double alpha, double beta, np.ndarray[np.float64_t,ndim=1] T, int n):

    cdef np.ndarray[np.float64_t,ndim=1] l = np.zeros(n, dtype=np.float64)
    
    cdef np.ndarray[np.float64_t,ndim=1] dT = T[1:]-T[:-1]
    cdef np.ndarray[np.float64_t,ndim=1] r = np.exp(-beta*dT)
    
    cdef double x = 0.0
    
    for i in xrange(n-1):
        x   = ( x   + alpha*beta  ) * r[i] 
        l[i+1] = x

    return l
    
    

