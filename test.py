import numpy as np
from scipy import sparse as sps
from scipy.sparse.linalg import spsolve # for finding exact solution
from scipy.io import loadmat
from matplotlib import pyplot
from cg import *
from cr import *

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

def testcase(A: sps.csc_matrix, b: np.ndarray, name: str, delta: float = 0.0) -> None:
    A, b = diag_precond(A, b)
    if delta != 0.0:
        name += "-indef"
        A = A - delta * sps.identity(A.shape[0])
    try:
        npz_file = np.load("matrices/" + name + ".npz")
        x_sol = npz_file["x_sol"]
        npz_file.close()
    except:
        print("Finding exact solution ...")
        x_sol = spsolve(A, b)
        np.savez("matrices/" + name + ".npz", x_sol = x_sol)
    if delta == 0.0:
        stat_cg = cg(A, b, x_sol)
    stat_cr = cr(A, b, delta != 0.0, x_sol)
    
    # r norm
    fig, ax = pyplot.subplots()
    if delta == 0.0:
        x_cg = np.arange(stat_cg.shape[0])
        ax.semilogy(x_cg, stat_cg[:,0], '-', label="CG")
    x_cr = np.arange(stat_cr.shape[0])
    ax.semilogy(x_cr, stat_cr[:,0], '-', label="MINRES")
    ax.set_xlabel("step")
    ax.set_ylabel("log |r|")
    ax.legend()
    pyplot.draw(); pyplot.pause(1e-4)
    fig.savefig("plots/" + name + "-4.eps", dpi=300.0)

    # x norm
    fig, ax = pyplot.subplots()
    if delta == 0.0:
        ax.plot(x_cg, stat_cg[:,1], '-', label="CG")
    ax.plot(x_cr, stat_cr[:,1], '-', label="MINRES")
    ax.set_xlabel("step")
    ax.set_ylabel("|x|")
    ax.legend()
    pyplot.draw(); pyplot.pause(1e-4)
    fig.savefig("plots/" + name + "-5.eps", dpi=300.0)

    if delta != 0.0:
        return

    # the backward error
    fig, ax = pyplot.subplots()
    ax.semilogy(x_cg, stat_cg[:,0] / stat_cg[:,1], '-', label="CG")
    ax.semilogy(x_cr, stat_cr[:,0] / stat_cr[:,1], '-', label="MINRES")
    ax.set_xlabel("step")
    ax.set_ylabel("log |r|/|x|")
    ax.legend()
    pyplot.draw(); pyplot.pause(1e-4)
    fig.savefig("plots/" + name + "-1.eps", dpi=300.0)

    # error 2-norm
    fig, ax = pyplot.subplots()
    ax.semilogy(x_cg, stat_cg[:,2], '-', label="CG")
    ax.semilogy(x_cr, stat_cr[:,2], '-', label="MINRES")
    ax.set_xlabel("step")
    ax.set_ylabel("log |x-x*|")
    ax.legend()
    pyplot.draw(); pyplot.pause(1e-4)
    fig.savefig("plots/" + name + "-2.eps", dpi=300.0)

    # error A-norm
    fig, ax = pyplot.subplots()
    ax.semilogy(x_cg, stat_cg[:,3], '-', label="CG")
    ax.semilogy(x_cr, stat_cr[:,3], '-', label="MINRES")
    ax.set_xlabel("step")
    ax.set_ylabel("log |x-x*|_A")
    ax.legend()
    pyplot.draw(); pyplot.pause(1e-4)
    fig.savefig("plots/" + name + "-3.eps", dpi=300.0)


if __name__ == "__main__":

    pyplot.ion()

    # ==========================================================
    # spd cases start

    print("\n* sts4098")
    mat = loadmat("matrices/sts4098.mat")
    struct = mat["Problem"][0,0]
    A = struct[2]
    b = np.asarray(struct[3].todense()).flatten()
    del mat
    testcase(A, b, "sts4098")

    print("\n* Simon_raefsky4")
    mat = loadmat("matrices/raefsky4.mat")
    struct = mat["Problem"][0,0]
    A = struct[1]
    b = struct[3].flatten()
    del mat
    testcase(A, b, "raefsky4")

    print("\n* Schmid_thermal1")
    mat = loadmat("matrices/thermal1.mat")
    struct = mat["Problem"][0,0]
    A = struct[2]
    b = struct[3].flatten()
    del mat
    testcase(A, b, "thermal1")

    # ==========================================================
    # indefinite cases start
    print("\n* sts4098, delta = 0.5")
    mat = loadmat("matrices/sts4098.mat")
    struct = mat["Problem"][0,0]
    A = struct[2]
    b = np.asarray(struct[3].todense()).flatten()
    del mat
    testcase(A, b, "sts4098", 0.5)

    print("\n* Schmid_thermal1, delta = 0.5")
    mat = loadmat("matrices/thermal1.mat")
    struct = mat["Problem"][0,0]
    A = struct[2]
    b = struct[3].flatten()
    del mat
    testcase(A, b, "thermal1", 0.5)
    
    print("\n* Simon_olafu, delta = 0.5")
    mat = loadmat("matrices/olafu.mat")
    struct = mat["Problem"][0,0]
    A = struct[1]
    b = struct[2].flatten()
    del mat
    testcase(A, b, "olafu", 0.5)

    pyplot.ioff()
    pyplot.show()
