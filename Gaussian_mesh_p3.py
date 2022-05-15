 #!/usr/bin/env python
import numpy as np
import math
from scipy.special.orthogonal import  p_roots

def Gaussian_mesh (npts_array, lower_limit_array, upper_limit_array):
    """ Gaussian_mesh provides nodes and weights for a specified type
    of gaussian quadrature.
 
    The inputs are:
    npts --- array with number of points to use for each interval
       lower_limit_array --- array with lower limits of integration defining intervals
       upper_limit_array --- array with upper limits of integration
 
    The outputs are:
        nodes --- row vector of npts gaussian nodes
        weights --- row vector of npts gaussian weights
     """
    
    nodes = np.array([])
    weigths = np.array([])
    
    for i in range(len(npts_array)):
        npts = npts_array[i]
        a = lower_limit_array[i]
        b = upper_limit_array[i]
        nodes_tmp, weights_tmp = Gaussian_quadrature (npts, a, b);
        nodes = np.append(nodes, nodes_tmp)
        weigths = np.append(weigths, weights_tmp)
    return nodes, weigths

def Gaussian_quadrature (npts, a, b):
    """ Gaussian_quadrature provides nodes and weights for a specified type
    of gaussian quadrature.
    
    The inputs are:
        npts --- number of points to use
        a --- lower limit of integration
        b --- upper limit of integration
    
    The outputs are:
        nodes --- row vector of npts gaussian nodes
        weights --- row vector of npts gaussian weights
        """
        
    eps = 3.e-12	# limit for accuracy (this is 3.e-10 in original)
    
    # initialize some variables
    t = 0.0; t1 = 0.0; p1 = 0.0; p2 = 0.0; p3 = 0.0; pp = 0.0;
    
    nodes, weights = p_roots(npts)
    
	# rescaling uniformly between (a,b) 
    nodes *=  (b - a) / 2.0
    nodes += (b + a) / 2.0
    weights *=  (b - a) / 2.0
    return nodes, weights

if __name__ == '__main__':
    from scipy.special.orthogonal import  p_roots
    npts_array = np.array([5, 40, 5])
    lower_limit_array = np.array([0, 13, 17])
    upper_limit_array = np.array([13, 17, 30])
    p, w = Gaussian_mesh(npts_array, lower_limit_array, upper_limit_array)
    def f(x):
        return np.exp(-(x-15)**2/0.01)
    from scipy.integrate import quad
    res = quad(f, 0, 30)[0]
    res2 = sum(f(p)*w)
    
    print (res -res2)

