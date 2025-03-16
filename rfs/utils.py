import scipy.io as sio
import numpy as np


def load_data(mat_file: str) -> tuple[np.ndarray, np.ndarray]:
    data = sio.loadmat(mat_file)
    return data["X"], data["Y"]


def calculate_centroids(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    unique_labels = np.unique(Y)
    unique_labels.sort()
    centriods = [
        np.mean(X[np.where(label == Y)[0], :], axis=0)
        for label in unique_labels
    ]
    return np.vstack(centriods)


if __name__ == "__main__":
    X, Y = load_data("data/ALLAML.mat")
    print(X.shape, Y.shape)