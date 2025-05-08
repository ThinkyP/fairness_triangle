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

def add_instance_dependent_noise(X, Y, w, b, max_flip_prob=0.5, seed=None):
    """
    Introduces instance-dependent label noise for binary labels {0, 1}.
    The closer a sample is to the decision boundary, the higher its chance of being flipped.

    Parameters:
    - X: array-like of shape (n_samples, 2), feature vectors
    - Y: array-like of shape (n_samples,), original labels in {0, 1}
    - w: weight vector of the linear boundary (e.g. from LDA)
    - b: bias term of the linear boundary
    - max_flip_prob: maximum flip probability at the decision boundary (typically 0.5)
    - seed: optional int, random seed for reproducibility

    Returns:
    - Y_corr: array with labels after instance-dependent noise corruption
    """
    if seed is not None:
        np.random.seed(seed)

    # Compute signed distances to the decision boundary
    signed_dists = np.array([
        compute_signed_distance_to_boundary(x[0], x[1], w, b) for x in X
    ])

    # Convert to absolute distances (smaller = closer to boundary)
    abs_dists = np.abs(signed_dists)

    # Normalize to [0, 1] and invert so that closer = higher flip prob
    max_dist = np.max(abs_dists)
    rel_proximity = 1 - abs_dists / max_dist

    # Scale to max_flip_prob
    flip_probs = rel_proximity * max_flip_prob

    # Perform stochastic flipping
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
