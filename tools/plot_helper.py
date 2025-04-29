import matplotlib.pyplot as plt
import scienceplots
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def science_fig(style="science", grid=True, scale=1.0, aspect_ratio=6/8, textwidth=7.285):
    """
    Creates a figure using scientific styles.

    Parameters:
    - style: str, which scienceplot style to use (default 'science')
    - grid: bool, whether to show grid
    - scale: scaling factor for figure size
    - aspect_ratio: figure height / width ratio
    - textwidth: figure width base size (inches)

    Returns:
    - fig: matplotlib figure object
    """
    styles = [style]
    if grid:
        styles.append("grid")

    plt.style.use(styles)

    width = textwidth * scale
    height = width * aspect_ratio

    plt.rcParams.update({
        'font.size': 15,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14
    })

    fig = plt.figure(figsize=(width, height))
    return fig



def truncate_colormap(cmap_name, minval=0.2, maxval=1.0, n=256):
    cmap = plt.get_cmap(cmap_name)
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap_name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

#Every function should work but this function is optimized for creating "scientific plots"
def plot_reg(Y_pred_p,Y_pred_f, Y_pred_all, xx, yy, X_s_0, y_s_0, X_s_1, y_s_1, disc_factor, a):
    textwidth =  7.285
    aspect_ratio = 6/8
    scale = 1.0
    width = textwidth * scale
    height = width * aspect_ratio
    
    plt.style.use(["science", "grid"])

    plt.rcParams.update({
    'font.size': 15,         # global font size
    'axes.labelsize': 15,    # axis label size
    'xtick.labelsize': 15,   # x-axis tick label size
    'ytick.labelsize': 15   # y-axis tick label size
    })
    plt.figure(figsize=(width, height))
    
    contour_p = plt.contour(xx, yy, Y_pred_p, levels=[0.5], linestyles='--', colors='green')
    contour_f = plt.contour(xx, yy, Y_pred_f, levels=[0.5], linestyles='--', colors='orange')
    contour_all = plt.contour(xx, yy, Y_pred_all, levels=[0], linestyles='--', colors='purple')
    
    
    
    #plt.scatter(X_s_0[y_s_0==1.0][:, 0], X_s_0[y_s_0==1.0][:, 1], color='green', marker='x', s=32, linewidth=1.4, label= "Prot. +ve disc_fac:")
    #plt.scatter(X_s_0[y_s_0==0][:, 0], X_s_0[y_s_0==0][:, 1], color='red', marker='x', s=32, linewidth=1.4, label = "Prot. -ve")
    #plt.scatter(X_s_1[y_s_1==1.0][:, 0], X_s_1[y_s_1==1.0][:, 1], color='lightgreen', marker='o', facecolors='none', s=30, label = "Non-prot. +ve")
    #plt.scatter(X_s_1[y_s_1==0][:, 0], X_s_1[y_s_1==0][:, 1], color='orange', marker='o', facecolors='none', s=30, label = "Non-prot. -ve")
    
    
    plt.scatter(X_s_0[y_s_0==1.0][:, 0], X_s_0[y_s_0==1.0][:, 1], color='green', marker='x', s=32, linewidth=1.4, label= "Non-Prot. +ve")
    plt.scatter(X_s_0[y_s_0==0][:, 0], X_s_0[y_s_0==0][:, 1], color='red', marker='x', s=32, linewidth=1.4, label = "Non-Prot. -ve")
    plt.scatter(X_s_1[y_s_1==1.0][:, 0], X_s_1[y_s_1==1.0][:, 1], color='lightgreen', marker='o', facecolors='none', s=30, label = "prot. +ve")
    plt.scatter(X_s_1[y_s_1==0][:, 0], X_s_1[y_s_1==0][:, 1], color='orange', marker='o', facecolors='none', s=30, label = "prot. -ve")
    


    legend = plt.legend()
    legend = plt.legend(fancybox=False, edgecolor="black")
    legend.get_frame().set_linewidth(0.5)

    handles, labels = legend.legendHandles, [text.get_text() for text in legend.get_texts()]
    handles_p, _ = contour_p.legend_elements()
    handles_f, _ = contour_f.legend_elements()
    handles_all, _ = contour_all.legend_elements()
    new_labels = [f'Disc. Factor: {disc_factor:.2f}'] + [r'$\lambda$: {:.1f}'.format(a)] + labels +   ['Perf. Reg'] + ['Fair. Reg'] + ['Fair. aware Reg']
    new_handles = [plt.Line2D([0], [0], color='black', linewidth=0)] + [plt.Line2D([0], [0], color='black', linewidth=0)] +  handles + handles_p + handles_f + handles_all
    
    plt.legend(new_handles, new_labels, loc=2, fontsize=12)
    
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off') # dont need the ticks to see the data distribution
    plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    plt.xlim((-15,10))
    plt.ylim((-10,15))
    plt.savefig(f"../img/E0_Synth_Data/data{disc_factor:.2f}_a_{a}.png")
    plt.show()
    

def subplot_reg(ax, Y_pred_p, Y_pred_f, Y_pred_all, xx, yy, X_s_0, y_s_0, X_s_1, y_s_1, disc_factor, a, legend_flag=True):
    contour_p = ax.contour(xx, yy, Y_pred_p, levels=[0.5], linestyles='--', linewidths = 1.2, colors='green')
    contour_f = ax.contour(xx, yy, Y_pred_f, levels=[0.5], linestyles='--', linewidths = 1.2, colors='orange')
    contour_all = ax.contour(xx, yy, Y_pred_all, levels=[0.5], linestyles='--', linewidths = 2, colors='purple')
    
    ax.scatter(X_s_0[y_s_0==1.0][:, 0], X_s_0[y_s_0==1.0][:, 1], color='green', marker='x', s=32, linewidth=1.4, label="Prot. +ve disc_fac:")
    ax.scatter(X_s_0[y_s_0==0][:, 0], X_s_0[y_s_0==0][:, 1], color='red', marker='x', s=32, linewidth=1.4, label="Prot. -ve")
    ax.scatter(X_s_1[y_s_1==1.0][:, 0], X_s_1[y_s_1==1.0][:, 1], color='lightgreen', marker='o', facecolors='none', s=30, label="Non-prot. +ve")
    ax.scatter(X_s_1[y_s_1==0][:, 0], X_s_1[y_s_1==0][:, 1], color='orange', marker='o', facecolors='none', s=30, label="Non-prot. -ve")
    
    legend = ax.legend()
    handles, labels = legend.legendHandles, [text.get_text() for text in legend.get_texts()]
    handles_p, _ = contour_p.legend_elements()
    handles_f, _ = contour_f.legend_elements()
    handles_all, _ = contour_all.legend_elements()
    new_labels = [f'disc_fact: {disc_factor:.2f}'] + [f'a: {a:.2f}'] + labels +  ['perf. reg'] + ['fair. reg'] + ['fair_aware reg']
    new_handles = [plt.Line2D([0], [0], color='black', linewidth=0)] + [plt.Line2D([0], [0], color='black', linewidth=0)] + handles + handles_p + handles_f + handles_all
    if legend_flag:
        ax.legend(new_handles, new_labels, loc=2, fontsize=10)
    else:
        ax.legend([plt.Line2D([0], [0], color='black', linewidth=0)] + handles, [f'a: {a:.2f}'] + labels, loc=2)
    ax.set_title(f"Synthetic Data - disc_factor{disc_factor:.2f}")
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)  # remove x-axis ticks
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)  # remove y-axis ticks

def subplot_reg_corr(fig, ax, Y_pred_p, Y_pred_p_cor, Y_pred_f, Y_pred_f_cor, Y_pred_all, xx, yy, X_s_0, y_s_0, X_s_1, y_s_1, disc_factor, a, legend_flag=True, legend_outside=False):
    contour_p = ax.contour(xx, yy, Y_pred_p, levels=[0.5], linestyles='--', linewidths = 1.2, colors='green')
    contour_p_cor = ax.contour(xx, yy, Y_pred_p_cor, levels=[0.5], linestyles='--', linewidths = 1.5, colors='lightgreen')
    contour_f = ax.contour(xx, yy, Y_pred_f, levels=[0.5], linestyles='--', linewidths = 1.2, colors='orange')
    contour_f_cor = ax.contour(xx, yy, Y_pred_f_cor, levels=[0.5], linestyles='--', linewidths = 1.5, colors='red')
    contour_all = ax.contour(xx, yy, Y_pred_all, levels=[0.5], linestyles='--', linewidths = 2, colors='purple')
    
    ax.scatter(X_s_0[y_s_0==1.0][:, 0], X_s_0[y_s_0==1.0][:, 1], color='green', marker='x', s=32, linewidth=1.4, label="Prot. +ve disc_fac:")
    ax.scatter(X_s_0[y_s_0==0][:, 0], X_s_0[y_s_0==0][:, 1], color='red', marker='x', s=32, linewidth=1.4, label="Prot. -ve")
    ax.scatter(X_s_1[y_s_1==1.0][:, 0], X_s_1[y_s_1==1.0][:, 1], color='lightgreen', marker='o', facecolors='none', s=30, label="Non-prot. +ve")
    ax.scatter(X_s_1[y_s_1==0][:, 0], X_s_1[y_s_1==0][:, 1], color='orange', marker='o', facecolors='none', s=30, label="Non-prot. -ve")
    
    legend = ax.legend()
    handles, labels = legend.legendHandles, [text.get_text() for text in legend.get_texts()]
    handles_p, _ = contour_p.legend_elements()
    handles_p_cor, _ = contour_p_cor.legend_elements()
    handles_f, _ = contour_f.legend_elements()
    handles_f_cor, _ = contour_f_cor.legend_elements()
    handles_all, _ = contour_all.legend_elements()
    
    manual_labels = [f'disc_fact: {disc_factor:.2f}'] + [f'a: {a:.2f}']
    manual_handles= [plt.Line2D([0], [0], color='black', linewidth=0)] + [plt.Line2D([0], [0], color='black', linewidth=0)]
    
    lines_labels= labels + ['perf. reg'] + ['corr_perf. reg'] + ['fair. reg'] + ['corr_fair. reg'] + ['fair_aware reg']
    lines_handles= handles + handles_p + handles_p_cor + handles_f + handles_f_cor + handles_all
    new_labels = manual_labels +  lines_labels
    new_handles =  manual_handles + lines_handles
    
    if legend_outside:
        fig.legend(lines_handles, lines_labels, loc='upper right', bbox_to_anchor=(0.5, 1.05), ncol=2)
        new_labels = manual_labels
        new_handles =  manual_handles
    if legend_flag:
        ax.legend(new_handles, new_labels, loc=2, fontsize=10)
    else:
        ax.legend([plt.Line2D([0], [0], color='black', linewidth=0)] + handles, [f'a: {a:.2f}'] + labels, loc=2)
    
        
    ax.set_title(f"Synthetic Data - disc_factor{disc_factor:.2f}")
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)  # remove x-axis ticks
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)  # remove y-axis ticks
    
    
    
    
    
    
def subplot_reg_corr(fig, ax, Y_pred_p, Y_pred_p_cor, Y_pred_f, Y_pred_f_cor, xx, yy, X_s_0, y_s_0, X_s_1, y_s_1, disc_factor, legend_flag=True, legend_outside=False):
    contour_p = ax.contour(xx, yy, Y_pred_p, levels=[0.5], linestyles='--', linewidths = 1.2, colors='green')
    contour_p_cor = ax.contour(xx, yy, Y_pred_p_cor, levels=[0.5], linestyles='--', linewidths = 1.5, colors='lightgreen')
    contour_f = ax.contour(xx, yy, Y_pred_f, levels=[0.5], linestyles='--', linewidths = 1.2, colors='orange')
    contour_f_cor = ax.contour(xx, yy, Y_pred_f_cor, levels=[0.5], linestyles='--', linewidths = 1.5, colors='red')
    
    ax.scatter(X_s_0[y_s_0==1.0][:, 0], X_s_0[y_s_0==1.0][:, 1], color='green', marker='x', s=32, linewidth=1.4, label="Prot. +ve")
    ax.scatter(X_s_0[y_s_0==0][:, 0], X_s_0[y_s_0==0][:, 1], color='red', marker='x', s=32, linewidth=1.4, label="Prot. -ve")
    ax.scatter(X_s_1[y_s_1==1.0][:, 0], X_s_1[y_s_1==1.0][:, 1], color='lightgreen', marker='o', facecolors='none', s=30, label="Non-prot. +ve")
    ax.scatter(X_s_1[y_s_1==0][:, 0], X_s_1[y_s_1==0][:, 1], color='orange', marker='o', facecolors='none', s=30, label="Non-prot. -ve")
    
    legend = ax.legend()
    handles, labels = legend.legendHandles, [text.get_text() for text in legend.get_texts()]
    handles_p, _ = contour_p.legend_elements()
    handles_p_cor, _ = contour_p_cor.legend_elements()
    handles_f, _ = contour_f.legend_elements()
    handles_f_cor, _ = contour_f_cor.legend_elements()
    
    manual_labels = [f'disc_fact: {disc_factor:.2f}']
    manual_handles= [plt.Line2D([0], [0], color='black', linewidth=0)]
    
    lines_labels= labels + ['perf. reg'] + ['corr_perf. reg'] + ['fair. reg'] + ['corr_fair. reg']
    lines_handles= handles + handles_p + handles_p_cor + handles_f + handles_f_cor 
    new_labels = manual_labels +  lines_labels
    new_handles =  manual_handles + lines_handles
    
    if legend_outside:
        fig.legend(lines_handles, lines_labels, loc='upper right', bbox_to_anchor=(0.5, 1.05), ncol=2)
        new_labels = manual_labels
        new_handles =  manual_handles
    if legend_flag:
        ax.legend(new_handles, new_labels, loc=2, fontsize=10)
    else:
        ax.legend([plt.Line2D([0], [0], color='black', linewidth=0)] + handles, [f'a: {a:.2f}'] + labels, loc=2)
    
        
    ax.set_title(f"Synthetic Data - disc_factor{disc_factor:.2f}")
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)  # remove x-axis ticks
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)  # remove y-axis ticks

