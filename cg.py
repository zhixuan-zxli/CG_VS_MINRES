import numpy as np
from scipy import sparse as spr

def cg(A: spr.csc_matrix, b: np.ndarray, indef: bool, x_sol: np.ndarray = 0.0, reltol: float = 1e-8) -> None:
    n, _ = A.shape
    b = b.reshape(-1)
    b_norm = np.linalg.norm(b, ord=None) # 2-norm
    nIter = n if indef else 5*n
    stats = np.zeros((nIter, 4)) # statistics with columns (r norm, x norm, err, err_A)
    # initial values
    x = np.zeros((n, ))
    r = b.copy()
    rho = r @ r
    p = r.copy()
    #
    for m in range(nIter):
        q = A @ p
        alpha = rho / (p@q)
        x += alpha * p
        r -= alpha * q
        rho_next = r @ r
        # save the statistics
        stats[m,0] = np.sqrt(rho_next)
        stats[m,1] = np.linalg.norm(x, ord=None)
        err = x - x_sol
        stats[m,2] = np.linalg.norm(err, ord=None)
        stats[m,3] = np.sqrt(err @ (A@err))
        # check for convergence
        if stats[m,0] <= reltol * b_norm:
            break
        # update the search direction
        beta = rho_next / rho
        p = r + beta * p
        rho = rho_next
    print("CG stops at iteration {} with relative residual {:.3e}".format(m, stats[m,0] / b_norm))
    return stats[:m+1]
