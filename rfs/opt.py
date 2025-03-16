import torch
import numpy as np


def opt_original_rfs(
    X: np.ndarray,
    Y: np.ndarray,
    l: float = 0.5,
    tol: float = 1e-3,
    max_iter: int = 1000,
    verbose: bool = False
) -> tuple[np.ndarray, list[float]]:
    d, n = X.shape
    A = np.hstack([X.T, l * np.eye(n)])
    D_inv = np.eye(n + d)
    losses = []
    for i in range(max_iter):
        # print(D_inv.shape, A.shape, Y.shape)
        U = D_inv @ A.T @ np.linalg.inv(A @ D_inv @ A.T) @ Y
        # print(U.shape)
        D_inv_new = np.diag(2 * np.linalg.norm(U, axis=1))
        loss = np.linalg.norm(U, ord=2, axis=1).sum()
        losses.append(loss)
        if verbose:
            # norm-2,1 of U
            print(f"Iteration {i + 1}: {loss}")
        if np.linalg.norm(D_inv_new - D_inv) < tol:
            break
        D_inv = D_inv_new
    return U[:d], losses    # return the first d rows of U: W


def conjugate_gradient_method(
    A: np.ndarray,
    Y: np.ndarray,
    X0: np.ndarray,
    tol: float = 1e-3,
    max_iter: int | None = None
) -> np.ndarray:
    # AX = Y
    # print(A.shape, Y.shape, X0.shape)

    max_iter = 2 * A.shape[0] if max_iter is None else max_iter

    X = X0.copy()

    for j in range(Y.shape[1]):
        x = X[:, j]
        y = Y[:, j]

        r = y - A @ x
        p = r.copy()
        rs_old = np.dot(r, r)

        for _ in range(max_iter):
            Ap = A @ p
            alpha = rs_old / np.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            rs_new = np.dot(r, r)

            if np.sqrt(rs_new) < tol:
                break

            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        X[:, j] = x

    return X


def opt_improved_rfs(
    X: np.ndarray,
    Y: np.ndarray,
    C_tilde: np.ndarray,
    l: float = 0.5,
    tol: float = 1e-3,
    max_iter: int = 1000,
    verbose: bool = False,
    cg: bool = False
) -> tuple[np.ndarray, list[float]]:
    d, n = X.shape
    k = Y.shape[1]
    A = np.hstack([X.T, l * np.eye(n)])
    D_inv = np.eye(n + d)
    losses = []
    for i in range(max_iter):
        # print(D_inv.shape, A.shape, Y.shape)
        # apply conjugate gradient method to solve ADA @ Y
        if cg:
            U = D_inv @ A.T @ conjugate_gradient_method(A @ D_inv @ A.T, Y, np.zeros((n, k))) @ C_tilde
        else:
            U = D_inv @ A.T @ np.linalg.inv(A @ D_inv @ A.T) @ Y @ C_tilde
        # print(U.shape)
        D_inv_new = np.diag(2 * np.linalg.norm(U, axis=1))
        loss = np.linalg.norm(U, ord=2, axis=1).sum()
        losses.append(loss)
        if verbose:
            # norm-2,1 of U
            print(f"Iteration {i + 1}: {loss}")
        if np.linalg.norm(D_inv_new - D_inv) < tol:
            break
        D_inv = D_inv_new
    return U[:d], losses    # return the first d rows of U: W