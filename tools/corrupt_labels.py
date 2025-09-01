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
    mask = np.random.rand(Y.shape[0]) <= flip_prob
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
    mask_0 = (Y == 0) & (np.random.rand(len(Y)) <= flip_prob_0_to_1)
    
    # Flip 1 → 0 with given probability
    mask_1 = (Y == 1) & (np.random.rand(len(Y)) <= flip_prob_1_to_0)
    
    Y_corr[mask_0] = 1    
    Y_corr[mask_1] = 0

    return Y_corr

def add_instance_dependent_noise(X, Y, w, b, max_flip_prob=0.5, alpha=1.0, seed=None):
    """
    Instance-dependent noise: samples near decision boundary are noisier.
    Flip probability decays exponentially with distance.
    """
    if seed is not None:
        np.random.seed(seed)

    # Signed distance (works for any dimension)
    signed_dists = (X @ w + b) / np.linalg.norm(w)
    abs_dists = np.abs(signed_dists)

    # Flip probability (exponential decay with distance)
    #flip_probs = max_flip_prob * np.exp(-alpha * abs_dists)
    
    # Normalize distances to [0,1]
    d_min, d_max = abs_dists.min(), abs_dists.max()
    norm_dists = (abs_dists - d_min) / (d_max - d_min + 1e-12)
    flip_probs = max_flip_prob * (1 - norm_dists)

    # Apply stochastic flipping
    random_vals = np.random.rand(len(Y))
    mask = random_vals < flip_probs
    Y_corr = np.where(mask, 1 - Y, Y)

    return Y_corr


def compute_signed_distance_to_boundary(x0, y0, w, b):
    """
    Computes the signed distance from point (x0, y0) to the decision boundary w^T x + b = 0.
    
    Positive if on Class 1 side, negative if on Class 0 side.
    """
    numerator = w[0] * x0 + w[1] * y0 + b
    denominator = np.linalg.norm(w)
    distance = numerator / denominator
    return distance
