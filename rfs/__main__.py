from . import opt, utils
import numpy as np
import time
import json


DATASETS = [
    ("rfs/data/ALLAML.mat", "rfs/stat/ALLAML.json"),
    ("rfs/data/GLIOMA.mat", "rfs/stat/GLIOMA.json"),
    ("rfs/data/lung.mat", "rfs/stat/lung.json")
]


X, Y = utils.load_data(DATASETS[0][0])
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
W2, l2 = opt.opt_improved_rfs(X, Y, C, max_iter=100, verbose=True)
t2 = time.time()
print("Improved RFS time: ", t2 - t1)
original_stat = {
    "time": t1 - t0,
    "losses": l1,
    "W": W1.tolist()
}
improved_stat = {
    "time": t2 - t1,
    "losses": l2,
    "W": W2.tolist()
}

# save the results
with open(DATASETS[0][1], "w") as f:
    json.dump({"original": l1, "improved": l2}, f, indent=2)