import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys

# Local tools
sys.path.insert(1, '/home/ptr@itd.local/code/fairness_triangle/tools')
from gen_synth_data import *
from plot_helper import *
from corrupt_labels import *
from calc_metrics import *

# Set global style and layout
plt.style.use(["science", "grid"])
st.set_page_config(layout="wide")


# --------------------------
# Sidebar Controls
# --------------------------
st.title("Label Noise Effects on Fairness and Performance")

flip_prob_0_to_1 = st.sidebar.slider('Asym: Flip 0 → 1', 0.0, 1.0, 0.3, step=0.01)
flip_prob_1_to_0 = st.sidebar.slider('Asym: Flip 1 → 0', 0.0, 1.0, 0.1, step=0.01)
sym_flip_prob = st.sidebar.slider('Sym: Flip Probability', 0.0, 1.0, 0.2, step=0.01)
max_inst_flip_prob = st.sidebar.slider('Instance-dependent Max Flip Prob', 0.0, 1.0, 0.5, step=0.01)


# Constants
rnd_seed = 0
disc_factor = np.pi / 2
n_samples = 2000
split_ratio = 0.7
c = c_bar = 0.1
lmd_start, lmd_end = 2, 15
lmd_interval = np.linspace(lmd_start, lmd_end, 30)
symmetric_fairness = True

st.sidebar.markdown("### Lambda Range")
lmd_start = st.sidebar.number_input('Lambda Start', min_value=-20.0, max_value=100.0, value=2.0, step=1.0)
lmd_end = st.sidebar.number_input('Lambda End', min_value=-20.0, max_value=100.0, value=15.0, step=1.0)
lmd_steps = st.sidebar.slider('Number of Lambda Steps', min_value=5, max_value=100, value=30, step=1)

# Ensure valid input
if lmd_end <= lmd_start:
    st.sidebar.error("Lambda End must be greater than Lambda Start.")
    st.stop()

lmd_interval = np.linspace(lmd_start, lmd_end, lmd_steps)

# --------------------------
# Caching Data Generation
# --------------------------

@st.cache_data
def get_synthetic_data(seed, disc_factor, n_samples):
    return generate_synthetic_data(False, n_samples, disc_factor, seed)

@st.cache_data
def get_asym_noise(Y, p_0_1, p_1_0, seed):
    return add_asym_noise(Y, p_0_1, p_1_0, seed)

@st.cache_data
def get_sym_noise(Y, flip_prob, seed):
    return add_sym_noise(Y, flip_prob, seed)

@st.cache_data
def get_instance_noise(X, Y, w, b, max_flip_prob, seed):
    return add_instance_dependent_noise(X, Y, w, b, max_flip_prob=max_flip_prob, seed=seed)

# Get base data
X, Y, Y_sen = get_synthetic_data(rnd_seed, disc_factor, n_samples)

# Boundary for instance-dependent noise
boundary_model = LogisticRegression().fit(X, Y)
w = boundary_model.coef_[0]
b = boundary_model.intercept_[0]

# Corrupt label variants
noise_versions = {
    "Symmetric Noise": get_sym_noise(Y, sym_flip_prob, rnd_seed),
    "Asymmetric Noise": get_asym_noise(Y, flip_prob_0_to_1, flip_prob_1_to_0, rnd_seed),
    "Instance-dependent Noise": get_instance_noise(X, Y, w, b, max_inst_flip_prob, rnd_seed)
}

# --------------------------
# Visualization Function
# --------------------------

def compute_and_plot(Y_corr, title):
    # Split
    split_index = int(split_ratio * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]
    Y_corr_train, Y_corr_test = Y_corr[:split_index], Y_corr[split_index:]
    Y_sen_train, Y_sen_test = Y_sen[:split_index], Y_sen[split_index:]

    # Train models
    p_reg = LogisticRegression().fit(X_train, Y_train)
    f_reg = LogisticRegression().fit(X_train, Y_sen_train)
    p_reg_corr = LogisticRegression().fit(X_train, Y_corr_train)
    f_reg_corr = LogisticRegression().fit(X_train, Y_sen_train)

    # Metrics
    BER_clean, MD_clean, DI_clean = [], [], []
    BER_corr, MD_corr, DI_corr = [], [], []

    for lmd in lmd_interval:
        s = p_reg.predict_proba(X_test)[:, 1] - c - lmd * (f_reg.predict_proba(X_test)[:, 1] - c_bar)
        Y_pred = np.where(s > 0, 1, 0)
        BER_clean.append(calc_BER(Y_pred, Y_test))
        MD_clean.append(calc_MD(Y_pred, Y_sen_test, symmetric_fairness))
        DI_clean.append(calc_DI(Y_pred, Y_sen_test, symmetric_fairness))

        s2 = p_reg_corr.predict_proba(X_test)[:, 1] - c - lmd * (f_reg_corr.predict_proba(X_test)[:, 1] - c_bar)
        Y_pred2 = np.where(s2 > 0, 1, 0)
        BER_corr.append(calc_BER(Y_pred2, Y_test))
        MD_corr.append(calc_MD(Y_pred2, Y_sen_test, symmetric_fairness))
        DI_corr.append(calc_DI(Y_pred2, Y_sen_test, symmetric_fairness))

    results = pd.DataFrame({
        'lambda': lmd_interval,
        'BER_clean': BER_clean,
        'MD_clean': MD_clean,
        'DI_clean': DI_clean,
        'BER_corr': BER_corr,
        'MD_corr': MD_corr,
        'DI_corr': DI_corr
    })

    # Plot
    fig = science_fig()
    ax = fig.gca()
    df_sorted = results.sort_values(by="lambda")
    lambda_norm = (results["lambda"] - results["lambda"].min()) / (results["lambda"].max() - results["lambda"].min())

    orange_cmap = truncate_colormap('Oranges', 0.2, 1.0)
    green_cmap = truncate_colormap('Greens', 0.2, 1.0)

    ax.plot(df_sorted["MD_clean"], df_sorted["BER_clean"], color="darkorange", label="Clean", linewidth=1)
    ax.plot(df_sorted["MD_corr"], df_sorted["BER_corr"], color="forestgreen", label="Noisy", linewidth=1)
    ax.scatter(results["MD_clean"], results["BER_clean"], c=lambda_norm, cmap=orange_cmap, s=60, alpha=0.6)
    ax.scatter(results["MD_corr"], results["BER_corr"], c=lambda_norm, cmap=green_cmap, s=60, alpha=0.6, marker='s')

    ax.set_xlabel("Mean Distance (MD)")
    ax.set_ylabel("Balanced Error Rate (BER)")
    ax.set_xlim(-0.55, 0.05)
    ax.set_ylim(0.2, 0.6)
    ax.legend()
    ax.set_title(title)
    st.pyplot(fig)

# --------------------------
# Display All Three Plots
# --------------------------
col1, col2, col3 = st.columns(3)

with col1:
    compute_and_plot(noise_versions["Symmetric Noise"], "Symmetric Noise")

with col2:
    compute_and_plot(noise_versions["Asymmetric Noise"], "Asymmetric Noise")

with col3:
    compute_and_plot(noise_versions["Instance-dependent Noise"], "Instance-dependent Noise")
