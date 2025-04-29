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

def calc_MD(Y_pred, Y_test, symmetrized=True):
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
