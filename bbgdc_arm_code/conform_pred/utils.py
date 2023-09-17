import numpy as np
from math import floor, ceil, factorial

def imethod(k, eta): # k : number of models
    k2c = int(ceil(k/2.0))
    k2f = int(floor(k/2.0))

    # n choose k
    choose = lambda n, k: factorial(n) / (factorial(k) * factorial(n-k))

    # determine the coefficients of the polynomial
    coefs = []
    for j in range(k2c, k+1):
        c = 0
        for i in range(k-j, k2f+1):
            c += choose(k, i) * choose(i, k-j) * (-1)**(j-k+i)
        coefs.append(c)
    
    # set remaining coefficients to 0, and the first to -eta
    coefs = [-eta] + [0]*(k-len(coefs)) + coefs
    # Find root in the interval [0, 1]
    for root in np.roots(coefs[::-1]):
        if root.imag != 0:
            continue
        if 0 <= root <= 1:
            return root.real
    raise Exception("no root was found")

