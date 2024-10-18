import numpy as np
from scipy import sparse as sps

def cr(A: sps.csc_matrix, b: np.ndarray, indef: bool, x_sol: np.ndarray = 0.0, reltol: float = 1e-8) -> np.ndarray:
    n = A.shape[0]
    b = b.reshape(-1)
    b_norm = np.linalg.norm(b, ord=None) # 2-norm
    maxIter = n if indef else 5*n
    stats = np.zeros((maxIter, 4)) # statistics with columns (r norm, x norm, err, err_A)
    # initial values
    x = np.zeros((n, ))
    r = b.copy()
    s = A @ r
    rho = r @ s
    p = r.copy()
    q = s.copy()
    
    for m in range(maxIter):
        # udpate the iterate and the residual
        alpha = rho / (q @ q)
        x += alpha * p
        r -= alpha * q
        #
        stats[m,0] = np.sqrt(r @ r)
        stats[m,1] = np.linalg.norm(x, ord=None)
        err = x - x_sol
        stats[m,2] = np.linalg.norm(err, ord=None)
        stats[m,3] = np.sqrt(err @ (A@err))
        # check for convergence
        if stats[m,0] <= reltol * b_norm:
            break
        # update the search direction
        s = A @ r
        rho_next = r @ s
        beta = rho_next / rho
        p = r + beta * p
        q = s + beta * q
        rho = rho_next

    print("CR stops at iteration {} with relative residual {:.3e}".format(m, stats[m,0] / b_norm))
    return stats[:m+1]
