# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys

# Your helper imports
sys.path.insert(1, '/home/ptr@itd.local/code/fairness_triangle/tools')  # Update if needed
from gen_synth_data import *
from plot_helper import *
from corrupt_labels import *
from calc_metrics import *

# Set global plot style
plt.style.use(["science", "grid"])

# --------------------------
# Streamlit App Start
# --------------------------

st.title("Label Noise Effect on Fairness and Performance")

# Sidebar sliders
flip_prob_0_to_1 = st.sidebar.slider('Flip Probability 0 → 1', 0.0, 1.0, 0.3, step=0.01)
flip_prob_1_to_0 = st.sidebar.slider('Flip Probability 1 → 0', 0.0, 1.0, 0.1, step=0.01)

rnd_seed = 0
disc_factor = np.pi/2
n_samples = 2000
split_ratio = 0.7
c = c_bar = 0.1
lmd_start = 2
lmd_end = 15
lmd_interval = np.linspace(lmd_start, lmd_end, 30)
symmetric_fairness = True

# Generate synthetic data
X, Y, Y_sen = generate_synthetic_data(False, n_samples, disc_factor, rnd_seed)
Y_corrupted = add_asym_noise(Y, flip_prob_0_to_1, flip_prob_1_to_0, rnd_seed)
Y_sen_corrupted = add_sym_noise(Y_sen, 0, rnd_seed)

# Split
split_index = int(split_ratio * len(X))
X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]
Y_corr_train, Y_corr_test = Y_corrupted[:split_index], Y_corrupted[split_index:]
Y_sen_train, Y_sen_test = Y_sen[:split_index], Y_sen[split_index:]
Y_sen_corr_train, Y_sen_corr_test = Y_sen_corrupted[:split_index], Y_sen_corrupted[split_index:]

# Train models
p_reg = LogisticRegression().fit(X_train, Y_train)
p_reg_cor = LogisticRegression().fit(X_train, Y_corr_train)
f_reg = LogisticRegression().fit(X_train, Y_sen_train)
f_reg_cor = LogisticRegression().fit(X_train, Y_sen_corr_train)

# Initialize lists
BER_list, MD_list, DI_list = [], [], []
BER_list_corr, MD_list_corr, DI_list_corr = [], [], []

# Sweep lambdas
for lmd in lmd_interval:
    s = p_reg.predict_proba(X_test)[:, 1] - c - lmd * (f_reg.predict_proba(X_test)[:, 1] - c_bar)
    Y_pred = np.where(s > 0, 1, 0)

    BER_list.append(calc_BER(Y_pred, Y_test))
    MD_list.append(calc_MD(Y_pred, Y_sen_test, symmetric_fairness))
    DI_list.append(calc_DI(Y_pred, Y_sen_test, symmetric_fairness))

for lmd in lmd_interval:
    s_2 = p_reg_cor.predict_proba(X_test)[:, 1] - c - lmd * (f_reg_cor.predict_proba(X_test)[:, 1] - c_bar)
    Y_pred_2 = np.where(s_2 > 0, 1, 0)

    BER_list_corr.append(calc_BER(Y_pred_2, Y_test))
    MD_list_corr.append(calc_MD(Y_pred_2, Y_sen_test, symmetric_fairness))
    DI_list_corr.append(calc_DI(Y_pred_2, Y_sen_test, symmetric_fairness))

# Create dataframe
results = pd.DataFrame({
    'lambda': lmd_interval,
    'BER_clean': BER_list,
    'MD_clean': MD_list,
    'DI_clean': DI_list,
    'BER_corr': BER_list_corr,
    'MD_corr': MD_list_corr,
    'DI_corr': DI_list_corr
})

# Plot
fig = science_fig()
ax = fig.gca()

df_sorted = results.sort_values(by="lambda")
lambda_norm = (results["lambda"] - results["lambda"].min()) / (results["lambda"].max() - results["lambda"].min())

orange_cmap = truncate_colormap('Oranges', 0.2, 1.0)
green_cmap = truncate_colormap('Greens', 0.2, 1.0)

ax.plot(df_sorted["MD_clean"], df_sorted["BER_clean"], color="darkorange", linestyle='-', linewidth=1)
ax.plot(df_sorted["MD_corr"], df_sorted["BER_corr"], color="forestgreen", linestyle='-', linewidth=1)
ax.scatter(results["MD_corr"], results["BER_corr"], c=lambda_norm, cmap=green_cmap, label="Corrupted", s=50, marker="s", alpha=0.5)
ax.scatter(results["MD_clean"], results["BER_clean"], c=lambda_norm, cmap=orange_cmap, label="Clean", linewidths=0.2, s=65)

ax.set_xlabel("MD (Mean Distance)")
ax.set_ylabel("Balanced Error")
ax.set_xlim(-0.55, 0.05)
ax.set_ylim(0.2, 0.6)
ax.legend(loc='upper left')
fig.tight_layout()

st.pyplot(fig)
