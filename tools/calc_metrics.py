import numpy as np
import matplotlib.pyplot as plt
from random import seed, shuffle
from scipy.stats import multivariate_normal 

SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)

def calc_confusion_matrix(Y_pred, Y_test):
    TP = np.sum((Y_pred == 1) & (Y_test == 1))  # True Positives
    TN = np.sum((Y_pred == 0) & (Y_test == 0))  # True Negatives
    FP = np.sum((Y_pred == 1) & (Y_test == 0))  # False Positives
    FN = np.sum((Y_pred == 0) & (Y_test == 1))  # False Negatives
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) != 0 else 0
    
    return TP, TN, FP, FN, FPR, FNR

def calc_BER(Y_pred, Y_test,):
    TP, TN, FP, FN, FPR, FNR = calc_confusion_matrix(Y_pred, Y_test)
    BER = (FPR + FNR) / 2
    return BER

def calc_MD(Y_pred, Y_test, symmetrized = True):
    #MD IS CALC USING Y_SEN!!!!
    TP, TN, FP, FN, FPR, FNR = calc_confusion_matrix(Y_pred, Y_test)
    MD = FPR + FNR - 1        # Y is 0 or 1
    
    if symmetrized:
        MD = -abs(MD)
    return MD

def calc_DI(Y_pred, Y_test, symmetrized = True):
    #DI IS CALC USING Y_SEN!!!!
    TP, TN, FP, FN, FPR, FNR = calc_confusion_matrix(Y_pred, Y_test)
    if FNR == 1:            #This cathes zero divition
        DI = 0
    else:
        DI = FPR / (1-FNR)
    
    if symmetrized:
        Y_pred = np.array([1-x for x in Y_pred])
        TP, TN, FP, FN, FPR, FNR = calc_confusion_matrix(Y_pred, Y_test)
        DI_2 = FPR / (1-FNR) 
        return min(DI, DI_2)       
        
    return DI