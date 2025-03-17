import numpy as np
import json


def select_features(
    W: np.ndarray,
    n_features: int = 20
):
    # choose the top n_features features from W and set the rest to zero
    W = W.copy()
    top_k_indices = np.argsort(np.linalg.norm(W, ord=2, axis=0))[-n_features:]
    mask = np.zeros_like(W)
    mask[:, top_k_indices] = 1
    W = W * mask
    return W


def eval_distance_to_centroids(
    X: np.ndarray,
    Y: np.ndarray,
    C: np.ndarray,
    W: np.ndarray
) -> float:
    # calculate the distance between the transformed data and the centroids
    distance = np.linalg.norm(X.T @ W - Y @ C, ord=2)
    return distance