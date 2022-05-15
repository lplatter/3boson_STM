#/usr/bin/env python3
import numpy as np
from scipy.optimize import brentq
import numba
import Gaussian_mesh_p3 as Gaussian_mesh

@numba.jit("f8(f8,f8,f8)")
def logfct(E,p,q):
    return np.log((p*p+q*q+p*q-E)/(p*p+q*q-p*q-E))/p/q

@numba.jit("f8(f8,f8,f8,f8,f8)")
def inhomo(E,p,k,H,L):
    return (logfct(E,p,k)+2.*H/Lambda/Lambda)

@numba.jit("f8(f8,f8,f8)")
def dimer_propagator(E,q,gamma):
    return (gamma+np.sqrt(-E+0.75*q*q))/(-gamma*gamma-E+0.75*q*q)
#    return 1./(-gamma+np.sqrt(-E+0.75*q*q))

@numba.jit("f8(f8,f8,f8,f8,f8,f8,f8)")
def kernel(E,p,q,gamma,rho,H,L):
    return 1/(1.-gamma*rho)*2./np.pi*q*q*dimer_propagator(E,q,gamma)*(logfct(E,p,q)+2.*H/L/L)

@numba.jit("f8[:](f8,f8,f8,f8[:],f8[:])")
def fred2(E,gamma,rho,absc,weights):
    k=absc[30]
    dim=absc.shape[0]
    matrix=np.zeros((dim,dim))
    inh_vector=np.zeros((dim))
    inh_vector=inhomo(E,absc,k)
    for i in range(dim):
        p=absc[i]
        for j in range(dim):
            q=absc[j]
            matrix[i,j]=(i==j)-weights[j]*kernel(E,p,q,gamma,rho,H,L)
            
    return np.linalg.solve(matrix,inh_vector)


@numba.jit("f8(f8,f8,f8,f8,f8,f8[:],f8[:])")
def determinant_real(E,gamma,rho,H,L,absc,weights):
    dim=absc.shape[0]
    matrix=np.zeros((dim,dim))
    for i in range(dim):
        p=absc[i]
        for j in range(dim):
            q=absc[j]
            matrix[i,j]=(i==j)-weights[j]*kernel(E,p,q,gamma,rho,H,L)/3.
    return np.linalg.det(matrix)

@numba.jit("f8(f8,f8,f8)")
def logfct(E,p,q):
    return np.log((p*p+q*q+p*q-E)/(p*p+q*q-p*q-E))/p/q

@numba.jit("f8(f8,f8,f8,f8,f8)")
def inhomo(E,p,k,H,L):
    return (logfct(E,p,k)+2.*H/Lambda/Lambda)

@numba.jit("f8(f8,f8,f8)")
def dimer_propagator(E,q,gamma):
    return (gamma+np.sqrt(-E+0.75*q*q))/(-gamma*gamma-E+0.75*q*q)
#    return 1./(-gamma+np.sqrt(-E+0.75*q*q))

@numba.jit("f8(f8,f8,f8,f8,f8,f8,f8)")
def kernel(E,p,q,gamma,rho,H,L):
    return 1/(1.-gamma*rho)*2./np.pi*q*q*dimer_propagator(E,q,gamma)*(logfct(E,p,q)+2.*H/L/L)

@numba.jit("f8[:](f8,f8,f8,f8[:],f8[:])")
def fred2(E,gamma,rho,absc,weights):
    k=absc[30]
    dim=absc.shape[0]
    matrix=np.zeros((dim,dim))
    inh_vector=np.zeros((dim))
    inh_vector=inhomo(E,absc,k)
    for i in range(dim):
        p=absc[i]
        for j in range(dim):
            q=absc[j]
            matrix[i,j]=(i==j)-weights[j]*kernel(E,p,q,gamma,rho,H,L)
            
    return np.linalg.solve(matrix,inh_vector)


@numba.jit("f8(f8,f8,f8,f8,f8,f8[:],f8[:])")
def determinant_real(E,gamma,rho,H,L,absc,weights):
    dim=absc.shape[0]
    matrix=np.zeros((dim,dim))
    for i in range(dim):
        p=absc[i]
        for j in range(dim):
            q=absc[j]
            matrix[i,j]=(i==j)-weights[j]*kernel(E,p,q,gamma,rho,H,L)/3.
    return np.linalg.det(matrix)

numba.jit("f8(f8,f8,f8,f8,int32,int32,int32)")
def bindenergy_run(gamma,rho,H,L,N1,N2,N3):
    npts_array = np.array([N1,N2,N3])
    lower_array = np.array([0.,np.log(0.05*L),0.8*L])
    upper_array = np.array([0.05*L,np.log(0.8*L),L])
    nGrid = sum(npts_array)
    absc_temp,weights_temp = Gaussian_mesh.Gaussian_mesh(npts_array, lower_array, upper_array)
    absc_temp=np.real(absc_temp)
    for j in range(N2):
        absc_temp[N1+j]=np.exp(absc_temp[N1+j])
        weights_temp[N1+j]=absc_temp[N1+j]*weights_temp[N1+j]
    b3=brentq(lambda x:determinant_real(x,gamma,rho,H,L,absc_temp,weights_temp),-100.,-1.00001)
    return b3

if __name__ == '__main__':
    b3 = bindenergy_run(1.,0.,0.,2.1,300,300,300)
    print('B3 = ',b3)
    
