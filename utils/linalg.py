import numpy as np


def symmetric_inv(A_inv):
    """
    Computes the inverse of a Hermitian and positive-definite matrix via Cholesky decomposition. (Hermitian matrices are the complex extension of real symmetric matrices.)
    The inverse of a symmetric matrix must also be symmetric. But np.linalg.inv(S_N_inv) does not necessarily preserve symmetry.
    As a remedy, compute the Cholesky decomposition, i.e., A = L * L.H, where L is lower-triangular and .H is the conjugate transpose operator.

    Cholesky decomposition will raise a LinAlgError if the decomposition fails, for example if the given matrix is not positive-definite.
    This code does not check that S_N_inv is in fact Hermitian.
    :param A_inv: Hermitian (symmetric for real-valued matrices) and positive-definite matrix for which to compute the inverse
    :return: A
    """

    L = np.linalg.cholesky(A_inv)
    # Compute inverse of L
    L_inv = np.linalg.inv(L)
    # Put together inverse of A_inv
    A = np.dot(L_inv.conj().T, L_inv)

    return A
