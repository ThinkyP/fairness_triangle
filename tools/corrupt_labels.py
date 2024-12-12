import numpy as np
import matplotlib.pyplot as plt
from random import seed, shuffle
from scipy.stats import multivariate_normal 

SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)

def add_bin_noise(Y, flip_prob):
    mask = np.random.rand(Y.shape[0]) < flip_prob
    Y_corr =  np.where(mask, 1 - Y, Y)         # Y is -1 or 1
    return Y_corr

