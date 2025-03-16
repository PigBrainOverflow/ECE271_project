from . import opt, utils
import numpy as np


DATASET = "rfs/data/ALLAML.mat"


X, Y = utils.load_data(DATASET)
X = X.T
# one-hot encoding for Y
k = len(np.unique(Y))
Y = np.eye(k)[Y.ravel() - 1]
# print(X.shape, Y.shape)
U = opt.opt_original_rfs(X, Y, verbose=True)