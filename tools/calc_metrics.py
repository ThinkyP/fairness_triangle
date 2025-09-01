import numpy as np
import matplotlib.pyplot as plt
from random import seed, shuffle
from scipy.stats import multivariate_normal 

# Set random seed for reproducibility
SEED = 1122334455
seed(SEED)
np.random.seed(SEED)

def calc_confusion_matrix(Y_pred, Y_test):
    """
    Computes the confusion matrix components and error rates.

    Parameters:
    - Y_pred: array-like, predicted binary labels {0, 1}
    - Y_test: array-like, true binary labels {0, 1}

    Returns:
    - TP: True Positives
    - TN: True Negatives
    - FP: False Positives
    - FN: False Negatives
    - FPR: False Positive Rate
    - FNR: False Negative Rate
    """
    TP = np.sum((Y_pred == 1) & (Y_test == 1))
    TN = np.sum((Y_pred == 0) & (Y_test == 0))
    FP = np.sum((Y_pred == 1) & (Y_test == 0))
    FN = np.sum((Y_pred == 0) & (Y_test == 1))
    
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0.0
    FNR = FN / (FN + TP) if (FN + TP) != 0 else 0.0
    
    return TP, TN, FP, FN, FPR, FNR

def calc_BER(Y_pred, Y_test):
    """
    Calculates the Balanced Error Rate (BER).

    Parameters:
    - Y_pred: array-like, predicted binary labels {0, 1}
    - Y_test: array-like, true binary labels {0, 1}

    Returns:
    - BER: Balanced Error Rate
    """
    _, _, _, _, FPR, FNR = calc_confusion_matrix(Y_pred, Y_test)
    BER = (FPR + FNR) / 2
    return BER


def calc_ACC(Y_pred, Y_test):
    """
    Calculates the Balanced Error Rate (BER).

    Parameters:
    - Y_pred: array-like, predicted binary labels {0, 1}
    - Y_test: array-like, true binary labels {0, 1}

    Returns:
    - BER: Balanced Error Rate
    """
    TP, TN, FP, FN, FPR, FNR = calc_confusion_matrix(Y_pred, Y_test)
    acc = (TP + TN) / (TP + TN + FP + FN)
    return acc


def calc_MD(Y_pred, Y_test, symmetrized=False):
    """
    Calculates the Mean Difference (MD) between error rates.

    Parameters:
    - Y_pred: array-like, predicted binary labels {0, 1}
    - Y_test: array-like, true binary labels {0, 1}
    - symmetrized: if True, MD is returned as a non-positive value

    Returns:
    - MD: Mean Difference
    """
    _, _, _, _, FPR, FNR = calc_confusion_matrix(Y_pred, Y_test)
    MD = FPR + FNR - 1

    if symmetrized:
        MD = -abs(MD)
    
    return MD

def calc_DI(Y_pred, Y_test, symmetrized=True):
    """
    Calculates the Disparate Impact (DI) based on confusion matrix components.

    Parameters:
    - Y_pred: array-like, predicted binary labels {0, 1}
    - Y_test: array-like, true binary labels {0, 1}
    - symmetrized: if True, calculates DI for both label flips and returns the minimum

    Returns:
    - DI: Disparate Impact score
    """
    _, _, _, _, FPR, FNR = calc_confusion_matrix(Y_pred, Y_test)
    
    if FNR == 1:
        DI = 0.0
    else:
        DI = FPR / (1 - FNR)
    
    if symmetrized:
        Y_pred_flipped = 1 - np.array(Y_pred)
        _, _, _, _, FPR_flipped, FNR_flipped = calc_confusion_matrix(Y_pred_flipped, Y_test)
        
        if FNR_flipped == 1:
            DI_flipped = 0.0
        else:
            DI_flipped = FPR_flipped / (1 - FNR_flipped)
        
        return min(DI, DI_flipped)

    return DI

import numpy as np

def tune_c_and_cbar_separately(
    p_reg, f_reg, 
    X_val, Y_val, Y_sen_val, 
    lmd, 
    symmetric_fairness=False,
    c_grid=None, cbar_grid=None, 
    init_c=0.5, init_cbar=0.5,
    max_iters=3, atol=1e-4
):
    """
    Separately tunes c and c_bar on validation:
      - c is chosen to minimize BER (holding c_bar fixed)
      - c_bar is chosen to minimize MD (holding c fixed)
    Optionally performs a few coordinate-descent passes.

    Returns:
        c_opt, cbar_opt, ber_opt, md_opt
    """
    # Precompute scores on validation
    p_val = p_reg.predict_proba(X_val)[:, 1]
    f_val = f_reg.predict_proba(X_val)[:, 1]

    # Reasonable default grids over [0,1]
    if c_grid is None:
        c_grid = np.linspace(0.0, 1.0, 1001)
    if cbar_grid is None:
        cbar_grid = np.linspace(0.0, 1.0, 1001)

    # Initialize
    c = float(init_c)
    c_bar = float(init_cbar)
    ber_opt = None
    md_opt = None

    for _ in range(max_iters):
        # --- Step 1: optimize c for BER given current c_bar ---
        best_ber = np.inf
        best_c = c
        for c_candidate in c_grid:
            s = p_val - c_candidate - lmd * (f_val - c_bar)
            y_hat = (s > 0).astype(int)
            ber = calc_BER(y_hat, Y_val)
            if ber < best_ber:
                best_ber = ber
                best_c = float(c_candidate)
        c = best_c
        ber_opt = best_ber

        # --- Step 2: optimize c_bar for MD given new c ---
        best_md = np.inf
        best_cbar = c_bar
        for cbar_candidate in cbar_grid:
            s = p_val - c - lmd * (f_val - cbar_candidate)
            y_hat = (s > 0).astype(int)
            md = calc_MD(y_hat, Y_sen_val, symmetric_fairness)
            if md < best_md:
                best_md = md
                best_cbar = float(cbar_candidate)
        # check convergence
        if np.isclose(best_cbar, c_bar, atol=atol) and np.isclose(best_c, c, atol=atol):
            c_bar = best_cbar
            md_opt = best_md
            break

        c_bar = best_cbar
        md_opt = best_md

    return c, c_bar, ber_opt, md_opt
