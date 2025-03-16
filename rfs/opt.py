import torch
import numpy as np


def opt_original_rfs(
    X: np.ndarray,
    Y: np.ndarray,
    l: float = 0.5,
    tol: float = 1e-3,
    max_iter: int = 100,
    verbose: bool = False
) -> np.ndarray:
    d, n = X.shape
    A = np.hstack([X.T, l * np.eye(n)])
    D_inv = np.eye(n + d)
    for i in range(max_iter):
        # print(D_inv.shape, A.shape, Y.shape)
        U = D_inv @ A.T @ np.linalg.inv(A @ D_inv @ A.T) @ Y
        # print(U.shape)
        D_inv_new = np.diag(2 * np.linalg.norm(U, axis=1))
        if verbose:
            # norm-2,1 of U
            loss = np.linalg.norm(U, ord=2, axis=1).sum()
            print(f"Iteration {i + 1}: {loss}")
        if np.linalg.norm(D_inv_new - D_inv) < tol:
            break
        D_inv = D_inv_new
    return U[:d]    # return the first d rows of U: W


def opt_improved_rfs(
    X: np.ndarray,
    Y: np.ndarray,
    C_tilde: np.ndarray,
    l: float = 0.5,
    tol: float = 1e-3,
    max_iter: int = 100,
    verbose: bool = False
) -> np.ndarray:
    d, n = X.shape
    A = np.hstack([X.T, l * np.eye(n)])
    D_inv = np.eye(n + d)
    for i in range(max_iter):
        # print(D_inv.shape, A.shape, Y.shape)
        U = D_inv @ A.T @ np.linalg.inv(A @ D_inv @ A.T) @ Y @ C_tilde
        # print(U.shape)
        D_inv_new = np.diag(2 * np.linalg.norm(U, axis=1))
        if verbose:
            # norm-2,1 of U
            loss = np.linalg.norm(U, ord=2, axis=1).sum()
            print(f"Iteration {i + 1}: {loss}")
        if np.linalg.norm(D_inv_new - D_inv) < tol:
            break
        D_inv = D_inv_new
    return U[:d]    # return the first d rows of U: W