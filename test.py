import numpy as np
from scipy import sparse as sps
from scipy.sparse.linalg import spsolve # for finding exact solution
from scipy.io import loadmat
from matplotlib import pyplot
from cg import *

def diag_precond(A: sps.csc_matrix, b: np.ndarray) -> tuple[sps.csc_matrix, np.ndarray]:
    # ei_lm = sps.linalg.eigs(A, k=1, which='LM') 
    # ei_sm = sps.linalg.eigs(A, k=1, which="SM") # never mind, fail to converge
    # print("Condition number = {:.3e}".format(ei[0] / ei[1]))
    d = A.diagonal()
    D = 1.0 / np.sqrt(d)
    A_new = sps.diags(D) @ A @ sps.diags(D)
    b_new = D * b
    b_new = b_new / np.linalg.norm(b_new, ord=None)
    return A_new, b_new

def plotStat(stat_cg: np.ndarray, stat_mr: np.ndarray) -> None:
    fig, ax = pyplot.subplots()


if __name__ == "__main__":

    # test case: sts4098
    print("* sts4098")
    mat = loadmat("matrices/sts4098.mat")
    struct = mat["Problem"][0,0]
    A = struct[2]
    b = np.asarray(struct[3].todense()).flatten()
    A, b = diag_precond(A, b)
    print("Finding exact solution ...")
    x_sol = spsolve(A, b)
    stats = cg(A, b, False, x_sol)
