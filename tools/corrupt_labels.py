import numpy as np

def add_sym_noise(Y, flip_prob, seed):
    """
    Introduces symmetric label noise for binary labels {0, 1}.
    Each label is flipped (0 → 1 or 1 → 0) with the same probability.

    Parameters:
    - Y: array-like, original labels in {0, 1}
    - flip_prob: probability of flipping each label
    - seed: random seed for reproducibility

    Returns:
    - Y_corr: array with labels after symmetric noise corruption
    """
    np.random.seed(seed)
    mask = np.random.rand(Y.shape[0]) < flip_prob
    Y_corr =  np.where(mask, 1 - Y, Y)         # Y is 1 or 0
    return Y_corr


def add_asym_noise(Y, flip_prob_0_to_1, flip_prob_1_to_0, seed):
    """
    Can be seen as LIN model coined in Aditya et al 2018.
    
    Introduces asymmetric label noise for binary labels {0, 1}.

    Parameters:
    - Y: array-like, labels in {0, 1}
    - flip_prob_0_to_1: probability of flipping 0 → 1
    - flip_prob_1_to_0: probability of flipping 1 → 0
    - seed: random seed for reproducibility

    Returns:
    - Y_noisy: array with corrupted labels
    """
    if seed is not None:
        np.random.seed(seed)
    Y_corr = Y.copy()

    # Flip 0 → 1 with given probability
    mask_0 = (Y == 0) & (np.random.rand(len(Y)) < flip_prob_0_to_1)
    Y_corr[mask_0] = 1

    # Flip 1 → 0 with given probability
    mask_1 = (Y == 1) & (np.random.rand(len(Y)) < flip_prob_1_to_0)
    Y_corr[mask_1] = 0

    return Y_corr