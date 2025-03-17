from . import opt, utils, eval
import numpy as np
import time
import json


DATASETS = [
    ("rfs/data/ALLAML.mat", "rfs/stat/ALLAML.json"),
    ("rfs/data/GLIOMA.mat", "rfs/stat/GLIOMA.json"),
    ("rfs/data/lung.mat", "rfs/stat/lung.json")
]


def run_opt(dataset: tuple[str, str]):
    X, Y = utils.load_data(dataset[0])
    C = utils.calculate_centroids(X, Y)
    X = X.T
    # one-hot encoding for Y
    k = len(np.unique(Y))
    Y = np.eye(k)[Y.ravel() - 1]
    # print(X.shape, Y.shape)
    t0 = time.time()
    W1, l1 = opt.opt_original_rfs(X, Y @ C, max_iter=100, verbose=True)
    t1 = time.time()
    print("Original RFS time: ", t1 - t0)
    W2, l2 = opt.opt_improved_rfs(X, Y, C, max_iter=100, verbose=True, cg=True)
    t2 = time.time()
    print("Improved RFS time: ", t2 - t1)
    original_stat = {
        "time": t1 - t0,
        "losses": l1,
        # "W": W1.tolist()
    }
    improved_stat = {
        "time": t2 - t1,
        "losses": l2,
        # "W": W2.tolist()
    }

    SW1 = eval.select_features(W1, 200)
    SW2 = eval.select_features(W2, 200)
    d1 = eval.eval_distance_to_centroids(X, Y, C, SW1)
    d2 = eval.eval_distance_to_centroids(X, Y, C, SW2)
    print(f"Original RFS distance: {d1}")
    print(f"Improved RFS distance: {d2}")

    # save the results
    with open(dataset[1], "w") as f:
        json.dump({
            "original": original_stat,
            "improved": improved_stat
        }, f, indent=4)


def run_eval(dataset: tuple[str, str]):
    X, Y = utils.load_data(dataset[0])
    C = utils.calculate_centroids(X, Y)
    X = X.T
    # one-hot encoding for Y
    k = len(np.unique(Y))
    Y = np.eye(k)[Y.ravel() - 1]
    with open(dataset[1], "r") as f:
        stat = json.load(f)
    W1 = np.array(stat["original"]["W"])
    W2 = np.array(stat["improved"]["W"])
    d1 = eval.eval_distance_to_centroids(X, Y, C, W1)
    d2 = eval.eval_distance_to_centroids(X, Y, C, W2)
    print(f"Original RFS distance: {d1}")
    print(f"Improved RFS distance: {d2}")


if __name__ == "__main__":
    run_opt(DATASETS[2])
    # run_eval(DATASETS[0])