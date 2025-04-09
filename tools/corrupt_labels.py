import numpy as np

def add_bin_noise(Y, flip_prob, seed):
    np.random.seed(seed)
    mask = np.random.rand(Y.shape[0]) < flip_prob
    Y_corr =  np.where(mask, 1 - Y, Y)         # Y is -1 or 1
    return Y_corr

