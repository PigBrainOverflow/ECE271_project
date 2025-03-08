import torch
import numpy as np


def optimize(
    X: np.ndarray,
    C: np.ndarray,
    l: float = 0.1,
    lr: float = 1e-3,
    max_iter: int = 1000,
    device: str = "cuda"
) -> np.ndarray:
    """
    minimize ||C - AA^TX||_F^2 + l||A||_1
    A
    """
    device = "cpu" if not torch.cuda.is_available() else device
    X: torch.Tensor = torch.tensor(X, dtype=torch.float64, device=device)
    A: torch.Tensor = torch.randn(C.shape[0], X.shape[0], dtype=torch.float64, requires_grad=True, device=device)
    C: torch.Tensor = torch.tensor(C, dtype=torch.float64, device=device)

    optimizer = torch.optim.Adam([A], lr=lr)

    for i in range(max_iter):
        optimizer.zero_grad()
        loss = torch.norm(C - A @ A.T @ X, p="fro")**2 + l * torch.sum(torch.norm(A, p=1, dim=1))
        loss.backward()
        print(f"Iteration {i + 1}, Loss: {loss.item()}")
        optimizer.step()

    return A.detach().numpy()