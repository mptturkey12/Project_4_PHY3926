import numpy as np

def make_tridiagonal(Nspace, b, d, a):                                                                 
    D = d*np.identity(Nspace)+a*np.diagflat(np.ones(Nspace-1),1)+b*np.diagflat(np.ones(Nspace-1),-1)
    D[Nspace-1, 0], D[0, Nspace-1] = a, b
    return D   

D = make_tridiagonal(5, 1, 2, 3)
print(D)