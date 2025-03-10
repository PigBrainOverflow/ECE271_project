import torch
import numpy as np


def optimize(
    X: np.ndarray,
    C: np.ndarray,
    l: float = 0.1,
    lr: float = 1e-3,
    max_iter: int = 100,
    device: str = "cuda",
    is_sparse: bool = False
) -> np.ndarray:
    """
    minimize ||C - AA^TX||_F^2 + l||A||_1
    A
    """
    device = "cpu" if not torch.cuda.is_available() else device
    X: torch.Tensor = torch.tensor(X, dtype=torch.float64, device=device)
    A: torch.Tensor = torch.randn(C.shape[0], X.shape[0], dtype=torch.float64, requires_grad=True, device=device)
    C: torch.Tensor = torch.tensor(C, dtype=torch.float64, device=device)
    if is_sparse:
        X = X.to_sparse()

    optimizer = torch.optim.Adam([A], lr=lr)

    for i in range(max_iter):
        optimizer.zero_grad()
        AAX = torch.sparse.mm(A @ A.T, X) if is_sparse else A @ A.T @ X
        loss = torch.norm(C - AAX, p="fro")**2 + l * torch.sum(torch.norm(A, p=1, dim=1))
        loss.backward()
        print(f"Iteration {i + 1}, Loss: {loss.item()}")
        optimizer.step()

    return A.detach().cpu().numpy()


def optimize_sparse(
    X: np.ndarray,
    C: np.ndarray,
    l: float = 0.1,
    lr: float = 1e-3,
    max_iter: int = 100,
    device: str = "cuda"
) -> np.ndarray:
    """
    minimize ||C - AX||_F^2 + l||A||_1
    A
    """
    device = "cpu" if not torch.cuda.is_available() else device
    X: torch.Tensor = torch.tensor(X, dtype=torch.float64, device=device)
    A: torch.Tensor = torch.randn(C.shape[0], X.shape[0], dtype=torch.float64, requires_grad=True, device=device)
    C: torch.Tensor = torch.tensor(C, dtype=torch.float64, device=device)

    optimizer = torch.optim.Adam([A], lr=lr)

    for i in range(max_iter):
        optimizer.zero_grad()
        loss = torch.norm(C - A @ X, p="fro")**2 + l * torch.sum(torch.norm(A, p=1, dim=1))
        loss.backward()
        print(f"Iteration {i + 1}, Loss: {loss.item()}")
        optimizer.step()

    return A.detach().cpu().numpy()


def select_features(
    A: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Select top k features (rows) from A and make the rest zero.
    """
    A = A.copy()
    top_k_indices = np.argsort(np.linalg.norm(A, ord=2, axis=0))[-k:]
    mask = np.zeros_like(A)
    mask[:, top_k_indices] = 1
    A = A * mask
    return A


def optimize_autoencoder(
    X: np.ndarray,
    Y: np.ndarray,
    We: np.ndarray | None = None,
    Wd: np.ndarray | None = None,
    n: int = 100,   # number of features
    l: float = 0.1,
    lr: float = 1e-3,
    max_iter: int = 100,
    device: str = "cuda",
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    minimize ||Y - Wd * We * X||_F^2 + l||We||_1
    """
    device = "cpu" if not torch.cuda.is_available() else device
    X: torch.Tensor = torch.tensor(X, dtype=torch.float64, device=device)
    Y: torch.Tensor = torch.tensor(Y, dtype=torch.float64, device=device)
    if We is None:
        We = torch.randn(n, X.shape[0], dtype=torch.float64, requires_grad=True, device=device)
    else:
        We = torch.tensor(We, dtype=torch.float64, requires_grad=True, device=device)
    if Wd is None:
        Wd = torch.randn(Y.shape[0], n, dtype=torch.float64, requires_grad=True, device=device)
    else:
        Wd = torch.tensor(Wd, dtype=torch.float64, requires_grad=True, device=device)

    optimizer = torch.optim.Adam([We, Wd], lr=lr)

    for i in range(max_iter):
        optimizer.zero_grad()
        loss = torch.norm(Y - Wd @ We @ X, p="fro")**2 + l * torch.sum(torch.norm(We, p=1, dim=1))
        loss.backward()
        if verbose:
            print(f"Iteration {i + 1}, Loss: {loss.item()}")
        optimizer.step()

    return We.detach().cpu().numpy(), Wd.detach().cpu().numpy()