import pickle
import numpy as np
import scipy.linalg as sp


def load_dataset(name: str, partition: int) -> tuple[np.ndarray, np.ndarray]:
    trainset = pickle.load(open(f"Data/{name}/Partition{partition}/train.p", "rb"))
    testset = pickle.load(open(f"Data/{name}/Partition{partition}/test.p", "rb"))
    return trainset, testset


def calculate_centroid(data: np.ndarray, labels: np.ndarray) -> dict[np.float64, np.ndarray]:
    unique_labels = np.unique(labels)
    centriods = {
        label: np.mean(data[np.where(label == labels)[0], :], axis=0)
        for label in unique_labels
    }
    return centriods


def calculate_C(labels: np.ndarray, centroids: dict[np.float64, np.ndarray]) -> np.ndarray:
    C = np.vstack([
        centroids[label]
        for label in labels
    ])
    return C


def solve_sylvester_sym(P, C):
    """
    Solves XP + PX = C for symmetric P and C.

    Args:
        P (ndarray): Symmetric matrix of shape (n, n).
        C (ndarray): Symmetric matrix of shape (n, n).

    Returns:
        X (ndarray): Solution matrix of shape (n, n).
    """
    # Step 1: Eigen decomposition of P
    eigvals, U = sp.eigh(P)  # P = UΛU^T, eigh ensures symmetry

    # Step 2: Transform C into eigenspace of P
    C_tilde = U.T @ C @ U  # U^T C U

    # Step 3: Solve for X in transformed space
    n = P.shape[0]
    X_tilde = np.zeros_like(C_tilde)

    for i in range(n):
        for j in range(n):
            denom = eigvals[i] + eigvals[j]
            if np.abs(denom) > 1e-12:  # Avoid division by zero
                X_tilde[i, j] = C_tilde[i, j] / denom

    # Step 4: Transform back to original space
    X = U @ X_tilde @ U.T  # X = U X_tilde U^T

    return X


if __name__ == "__main__":
    trainset, testset = load_dataset("ALLAML", 2)
    data, labels = trainset[:, :-1], trainset[:, -1]
    centroids = calculate_centroid(data, labels)
    X = data.T
    C = calculate_C(labels, centroids).T

    XXT = X @ X.T
    CXT = C @ X.T
    AAT = solve_sylvester_sym(P=XXT, C=CXT)
    print(AAT)
    print(AAT.shape)
    print(f"Residual Error:\n{XXT @ AAT + AAT @ XXT - CXT}")