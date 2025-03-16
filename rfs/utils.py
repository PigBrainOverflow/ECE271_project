import scipy.io as sio
import numpy as np


def load_data(mat_file: str) -> tuple[np.ndarray, np.ndarray]:
    data = sio.loadmat(mat_file)
    return data["X"], data["Y"]


if __name__ == "__main__":
    X, Y = load_data("data/ALLAML.mat")
    print(X.shape, Y.shape)