import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal 




def generate_synthetic_data(plot_data=False, n_samples=2000, disc_factor=np.pi / 4.0, seed=0):
    """
    Generate synthetic data for binary classification with a sensitive attribute.
    
    Parameters:
    - plot_data (bool): Whether to visualize the dataset.
    - n_samples (int): Samples per class.
    - disc_factor (float): Rotation angle for sensitive feature generation.
    - seed (int): Random seed for reproducibility.
    
    Returns:
    - X (ndarray): Feature matrix.
    - Y (ndarray): Target labels.
    - Y_sen (ndarray): Sensitive attribute.
    """
    n_samples = n_samples
    disc_factor = disc_factor
    np.random.seed(seed)
    
    def gen_gaussian(mean_in, cov_in, class_label):
    
        nv = multivariate_normal(mean = mean_in, cov = cov_in)
        X = nv.rvs(n_samples)
        y = np.ones(n_samples, dtype=float) * class_label
        return nv,X,y

    #p(x|y=1)
    mean1 = [2, 2]
    covar1 = [[5, 1], 
            [1, 5]]

    #p(x|y=0)
    mean2 = [-2, -2]
    covar2 = [[10, 1], 
            [1, 3]]

    nv1, Xp, Yp = gen_gaussian(mean1, covar1, 1)
    nv2, Xn, Yn = gen_gaussian(mean2, covar2, 0)

    X = np.vstack((Xp, Xn))
    Y = np.hstack((Yp, Yn))

    # shuffle the data
    perm = np.random.RandomState(seed=seed).permutation(X.shape[0])
    X = X[perm]
    Y= Y[perm]

    X_prime = X @ np.array([[np.cos(disc_factor), -np.sin(disc_factor)], [np.sin(disc_factor), np.cos(disc_factor)]])

    """ Generate the sensitive feature here """
    Y_sen = [] # this array holds the sensitive feature value
    for i in range (0, len(X)):
        x = X_prime[i]

        # probability for each cluster that the point belongs to it
        p1 = nv1.pdf(x)
        p2 = nv2.pdf(x)
        
        # normalize the probabilities from 0 to 1
        s = p1+p2
        p1 = p1/s
        p2 = p2/s
        
        r = np.random.uniform() # generate a random number from 0 to 1

        if r < p1: # the first cluster is the positive class
            Y_sen.append(1.0) # 1.0 means its male
        else:
            Y_sen.append(0.0) # 0.0 -> female

    Y_sen = np.array(Y_sen)

    """ Show the data """
    if plot_data:
        num_to_draw = 200 # we will only draw a small number of points to avoid clutter
        x_draw = X[:num_to_draw]
        y_draw = Y[:num_to_draw]
        Y_sen_draw = Y_sen[:num_to_draw]

        X_s_0 = x_draw[Y_sen_draw == 0.0]
        X_s_1 = x_draw[Y_sen_draw == 1.0]
        
        y_s_0 = y_draw[Y_sen_draw == 0.0]
        y_s_1 = y_draw[Y_sen_draw == 1.0]

        plt.scatter(X_s_0[y_s_0==1.0][:, 0], X_s_0[y_s_0==1.0][:, 1], color='green', marker='x', s=35, linewidth=1.5, label= "Non-Prot. +ve")
        plt.scatter(X_s_0[y_s_0==0][:, 0], X_s_0[y_s_0==0][:, 1], color='red', marker='x', s=35, linewidth=1.5, label = "Non-Prot. -ve")
        plt.scatter(X_s_1[y_s_1==1.0][:, 0], X_s_1[y_s_1==1.0][:, 1], color='green', marker='o', facecolors='none', s=30, label = "prot. +ve")
        plt.scatter(X_s_1[y_s_1==0][:, 0], X_s_1[y_s_1==0][:, 1], color='red', marker='o', facecolors='none', s=30, label = "prot. -ve")

        
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off') # dont need the ticks to see the data distribution
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
        plt.legend(loc=2, fontsize=15)
        plt.xlim((-15,10))
        plt.ylim((-10,15))
        #plt.savefig("img/data.png")
        plt.show()

   #Y_sen_draw = {"s1": Y_sen} # all the sensitive features are stored in a dictionary
    return X, Y, Y_sen