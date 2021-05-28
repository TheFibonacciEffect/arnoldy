import numpy as np
import scipy as sp
import scipy.linalg as spl

def get_largest(A):
    n = 50
    dim = A.shape[0]
    guess = np.random.rand(dim)
    v = np.linalg.matrix_power(A, n) @ guess
    return v / np.linalg.norm(v)

def get_smallest(A):
    # es gibt probleme, falls einer der eigenwerte 0 ist
    if spl.det(A) == 0:
        V, K, Vt = spl.svd(A)
        return V[:,-1]

    n = 50
    LUP = spl.lu_factor(A)
    dim = A.shape[0]
    guess = np.random.rand(dim)
    v = spl.lu_solve(LUP, guess)
    for i in range(n):
        v = spl.lu_solve(LUP, v) 
        v = v/ np.linalg.norm(v)
    return v

def eigval(A, v):
    return v.T @ A @ v / (v.T @ v)

def eig(A):
    assert A.shape[0] == A.shape[1] #square matrix
    v_max = get_largest(A)
    v_min = get_smallest(A)
    print(v_max)
    print(v_min)
    print(eigval(A, v_max))
    print(eigval(A, v_min))
    l_max = eigval(A, v_max)
    l_min = eigval(A, v_min)
    guesses = 30
    vals = np.linspace(l_min, l_max, guesses)
    dim = A.shape[0]
    # vects = np.zeros(dim-2,dim)
    vects = np.zeros((guesses,dim))
    for k, l in enumerate(vals):
        # This only returns the biggest and smallest eigenvalue, not the ones in between
        vects[k] = get_smallest(A - l*np.eye(dim))
        vals[k] = eigval(A, vects[k])
    print(f"eigvals {vals}")
    

def gram_schmidt(*vects):
    for n,v in enumerate(vects):
        for w in vects[:n]:
            v -= np.vdot(v,w) / np.vdot(w,w) * w
    return vects

if __name__ == "__main__":
    A = np.array([
        [1,0,1],
        [0,2,0],
        [0,0,3]
    ])
    eig(A)

    w = np.array([1.,0.,0.])
    v = np.array([1.,1.,0.])
    u = np.array([1.,1.,3.])
    v,w,u = gram_schmidt(v,w,u)
    print(v.T @ u)