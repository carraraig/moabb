import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import scipy
from pyriemann.utils.base_siegel import expm, invsqrtm, logm, powm, sqrtm, tanhm, arctanhm
from pyriemann.utils.covariance import covariances

class SiegelCoefficient(BaseEstimator, TransformerMixin):
    """
        Average the blocks over each diagonal for a block Toeplitz matrix.

        Parameters:
        - X: ACM Toeplitz matrix (epoch, (n * d) x (n * d)).
        - size_cov: The size of each block.
        - order: The number of blocks.

        Returns:
        - The matrix after averaging the blocks over each diagonal.
        """
    def __init__(self, order=1, lag=1, estimator="cov", **kwds):

        self.order = order
        self.lag = lag
        self.estimator = estimator
        self.kwds = kwds


    def fit(self, X, y):

        return self

    def transform(self, X):

        X_aug = augmented_dataset(X, order=self.order, lag=self.lag)

        covmats = covariances(X_aug, estimator=self.estimator, **self.kwds)

        self.size_cov = int(covmats.shape[1] / self.order)

        # Average over diagonal block to create a Block Toeplitz SPD
        Block_Toepliz = block_toeplitz_equal_diagonal_epoch(covmats, n=self.size_cov, d=self.order)

        # Compute the coeffiecient (P_0, Omega_0, ...., Omega_(n-1))
        Coeff = compute_HPD_and_Siegel_coefficients(Block_Toepliz, n=self.size_cov, d=self.order)

        return Coeff


def augmented_dataset(X, order, lag):
    if order == 1:
        X_fin = X
    else:
        X_p = X[:, :, : -order * lag]
        X_p = np.concatenate(
            [X_p]
            + [
                X[:, :, p * lag: -(order - p) * lag]
                for p in range(1, order)
            ],
            axis=1,
        )
        X_fin = X_p

    return X_fin

def block_toeplitz_equal_diagonal_epoch(matrix, n, d):
    """
    Average the blocks over each diagonal for a block Toeplitz matrix.

    Parameters:
    - matrix: The block Toeplitz matrix (n * d) x (n * d).
    - n: The size of each block.
    - d: The number of blocks.

    Returns:
    - The matrix after averaging the blocks over each diagonal.
    """
    epoch_matrix = np.zeros_like(matrix, dtype=float)

    for ep in np.arange(matrix.shape[0]):
        if matrix[ep].shape != (n * d, n * d):
            raise ValueError("Input matrix dimensions do not match the block Toeplitz structure.")

        result_matrix = np.zeros_like(matrix[ep], dtype=float)

        for delta in range(-d + 1, d):
            count = 0
            avg_value = np.zeros((n, n), dtype=float)

            for i in range(max(0, -delta), min(d, d - delta)):
                j = i + delta
                avg_value += matrix[ep, i * n:(i + 1) * n, j * n:(j + 1) * n]
                count += 1

            avg_value /= count

            for i in range(max(0, -delta), min(d, d - delta)):
                j = i + delta
                result_matrix[i * n:(i + 1) * n, j * n:(j + 1) * n] = avg_value

        epoch_matrix[ep, :, :] = result_matrix

    return epoch_matrix


def extract_autocorrelation_coefficients(R, n, d):
    autocorrelation_coefficients = [R[0: n, n*i:n*i+n] for i in range(d)]
    return autocorrelation_coefficients

def compute_HPD_and_Siegel_coefficients(R, n, d):

    Omega_Epoch = []

    for ep in np.arange(R.shape[0]):
        autocorrelation_coefficients = extract_autocorrelation_coefficients(R[ep], n, d)
        len_autoc = len(autocorrelation_coefficients) - 1

        # Initialization
        P_0 = autocorrelation_coefficients[0]
        Omega = []
        Omega.append(P_0)

        for l in range(0, len_autoc):
            if l == 0:
                term_1 = invsqrtm(np.array(autocorrelation_coefficients[0]))
                Omega_l = term_1 @ \
                          np.array(autocorrelation_coefficients[1]) @ term_1
                Omega.append(Omega_l)

            else:
                R_tilde_inv = np.linalg.inv(R[ep, 0:n*l, 0:n*l])
                R_l = np.concatenate([autocorrelation_coefficients[1 + i] for i in range(l)], axis=1)
                R_l_H = np.conj(R_l).T
                R_l_H_int = np.concatenate(
                    [np.conj(np.array(autocorrelation_coefficients[1 + i])).T for i in range(l)][::-1], axis=1)
                R_l_H_int_H = np.conj(R_l_H_int).T

                # Calculate L_l
                L_l = P_0 - R_l @ R_tilde_inv @ R_l_H

                # Calculate K_l
                K_l = P_0 - R_l_H_int @ R_tilde_inv @ R_l_H_int_H

                # Calculate M_l
                M_l = R_l @ R_tilde_inv @ R_l_H_int_H

                # Calculate Omega_{l+1}
                term_1 = invsqrtm(L_l)
                term_2 = invsqrtm(K_l)
                Omega_l = term_1 @ (np.array(autocorrelation_coefficients[l+1]) - M_l) @ term_2
                Omega.append(Omega_l)

        if not check_symmetric(P_0) or not np.all(scipy.linalg.eigh(P_0, eigvals_only=True) > 0):
            raise ValueError("P_0 is NOT belong to SPD")

        for k in np.arange(1, len(Omega)):
            M = Omega[k] @ np.conj(Omega[k]).T
            lambda_, _ = np.linalg.eigh(M)
            if not np.all(lambda_ <= 1 + 1e-12):
                raise ValueError("Coefficient NOT belong to Siegel Disk")

        Omega_Epoch.append(Omega)

    Omega_Final = np.array(Omega_Epoch)

    return Omega_Final

def is_pos_def(matrix_coeff):
    """
    Function that return True if the matrix is a Positive defined matrix
    :param matrix_coeff: matrix to check, have to be square
    :return:
        True if matrix is a SPD
    """
    if check_symmetric(matrix_coeff) & np.all(scipy.linalg.eigh(matrix_coeff, eigvals_only=True) > 0):
        print("The coefficient Matrix is a SPD matrix")
    else:
        print("The coefficient Matrix is NOT a SPD matrix")
    if not check_symmetric(matrix_coeff):
        print("The coefficient Matrix is not Symmetric")
    if not np.all(np.linalg.eigvals(matrix_coeff) > 0):
        print("Eigenvalue Not Positive")


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def check_siegel(a):
    M = np.conj(a).T @ a
    lambda_, _ = np.linalg.eigh(M)
    if np.all(lambda_ <= 1 + 1e-12):
        print("Coefficient Belong to Siegel Disk")
    else:
        print("Coefficient NOT belong to Siegel Disk")



