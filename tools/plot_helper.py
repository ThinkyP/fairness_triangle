import matplotlib.pyplot as plt
import scienceplots
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.style.use(["science", "grid"])
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.7
})

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

def plot_reg(Y_pred_p, Y_pred_f, Y_pred_all, 
             xx, yy, 
             X_s_0, y_s_0, X_s_1, y_s_1, 
             disc_factor=None, a=None):
    
    import matplotlib.pyplot as plt

    textwidth = 7.285
    aspect_ratio = 6/8
    scale = 1.0
    width = textwidth * scale
    height = width * aspect_ratio
    

    plt.figure(figsize=(width, height))

    # Plot contour lines
    contour_p = plt.contour(xx, yy, Y_pred_p, levels=[0.5], linestyles='--', colors='green')
    contour_f = plt.contour(xx, yy, Y_pred_f, levels=[0.5], linestyles='--', colors='orange')
    contour_all = plt.contour(xx, yy, Y_pred_all, levels=[0], linestyles='--', colors='purple')

    # Scatter data points
    plt.scatter(X_s_0[y_s_0 == 1.0][:, 0], X_s_0[y_s_0 == 1.0][:, 1], 
                color='green', marker='x', s=32, linewidth=1.4, label=r"$Y=1, \bar{Y}=0$")
    plt.scatter(X_s_0[y_s_0 == 0][:, 0], X_s_0[y_s_0 == 0][:, 1], 
                color='red', marker='x', s=32, linewidth=1.4, label=r"$Y=0, \bar{Y}=0$")
    plt.scatter(X_s_1[y_s_1 == 1.0][:, 0], X_s_1[y_s_1 == 1.0][:, 1], 
                color='lightgreen', marker='o', facecolors='none', s=30, label=r"$Y=1, \bar{Y}=1$")
    plt.scatter(X_s_1[y_s_1 == 0][:, 0], X_s_1[y_s_1 == 0][:, 1], 
                color='orange', marker='o', facecolors='none', s=30, label=r"$Y=0, \bar{Y}=1$")

    # Build the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    handles_p, _ = contour_p.legend_elements()
    handles_f, _ = contour_f.legend_elements()
    handles_all, _ = contour_all.legend_elements()

    new_labels = labels + ['Perf. Reg', 'Fair. Reg', 'Fair-aware Reg']
    new_handles = handles + handles_p + handles_f + handles_all

    # Add disc_factor and lambda if available
    if a is not None:
        new_labels.append(rf'$\lambda$: {a:.1f}')
        new_handles.append(plt.Line2D([0], [0], color='black', linewidth=0))
    if disc_factor is not None:
        new_labels.append(f'Disc. Factor: {disc_factor:.2f}')
        new_handles.append(plt.Line2D([0], [0], color='black', linewidth=0))

    plt.legend(new_handles, new_labels, loc=2, fontsize=12, frameon=True).get_frame().set_linewidth(0.5)

    # Axes cleanup
    plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    plt.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    plt.xlim((-15, 10))
    plt.ylim((-10, 15))

    # Save plot if both values are present
    if disc_factor is not None and a is not None:
        plt.savefig(f"../img/E0_Synth_Data/data{disc_factor:.2f}_a_{a}.png")
    
    plt.show()

    
def plot_reg_without_y_pred(Y_pred_p, Y_pred_f, 
             xx, yy, 
             X_s_0, y_s_0, X_s_1, y_s_1, 
             disc_factor=None, a=None):
    
    import matplotlib.pyplot as plt

    textwidth = 7.285
    aspect_ratio = 6 / 8
    scale = 1.0
    width = textwidth * scale
    height = width * aspect_ratio

    plt.figure(figsize=(width, height))

    # Plot contour lines
    contour_p = plt.contour(xx, yy, Y_pred_p, levels=[0.5], linestyles='--', colors='green')
    contour_f = plt.contour(xx, yy, Y_pred_f, levels=[0.5], linestyles='--', colors='orange')

    # Scatter data points
    plt.scatter(X_s_0[y_s_0 == 1.0][:, 0], X_s_0[y_s_0 == 1.0][:, 1], 
                color='green', marker='x', s=32, linewidth=1.4, label=r"$Y=1, \bar{Y}=0$")
    plt.scatter(X_s_0[y_s_0 == 0][:, 0], X_s_0[y_s_0 == 0][:, 1], 
                color='red', marker='x', s=32, linewidth=1.4, label=r"$Y=0, \bar{Y}=0$")
    plt.scatter(X_s_1[y_s_1 == 1.0][:, 0], X_s_1[y_s_1 == 1.0][:, 1], 
                color='lightgreen', marker='o', facecolors='none', s=30, label=r"$Y=1, \bar{Y}=1$")
    plt.scatter(X_s_1[y_s_1 == 0][:, 0], X_s_1[y_s_1 == 0][:, 1], 
                color='orange', marker='o', facecolors='none', s=30, label=r"$Y=0, \bar{Y}=1$")

    # Build the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    handles_p, _ = contour_p.legend_elements()
    handles_f, _ = contour_f.legend_elements()

    new_labels = labels + ['Perf. Reg', 'Fair. Reg']
    new_handles = handles + handles_p + handles_f

    # Add lambda and disc_factor if provided
    if a is not None:
        new_labels.append(rf'$\lambda$: {a:.1f}')
        new_handles.append(plt.Line2D([0], [0], color='black', linewidth=0))
    if disc_factor is not None:
        new_labels.append(f'Disc. Factor: {disc_factor:.2f}')
        new_handles.append(plt.Line2D([0], [0], color='black', linewidth=0))

    plt.legend(new_handles, new_labels, loc=2, fontsize=12, frameon=True).get_frame().set_linewidth(0.5)

    # Axes cleanup
    plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    plt.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    plt.xlim((-15, 10))
    plt.ylim((-10, 15))
    

    

def subplot_reg(ax, 
                Y_pred_p, Y_pred_f, Y_pred_all, 
                xx, yy, 
                X_s_0, y_s_0, X_s_1, y_s_1, 
                disc_factor, a, legend_flag=True,):
    """
    Draw decision boundaries and scatter points into a given Axes (ax).
    """

    # --- Contours ---
    contour_p   = ax.contour(xx, yy, Y_pred_p,   levels=[0.5], linestyles='--', linewidths=1.2, colors='green')
    contour_f   = ax.contour(xx, yy, Y_pred_f,   levels=[0.5], linestyles='--', linewidths=1.2, colors='orange')
    contour_all = ax.contour(xx, yy, Y_pred_all, levels=[0],   linestyles='--', linewidths=2,   colors='purple')

    # --- Data points ---
    # Y = 0
    ax.scatter(X_s_0[y_s_0 == 0][:, 0], X_s_0[y_s_0 == 0][:, 1],
            color='red', marker='x', s=32, linewidth=1.4, label=r"$Y=0, \bar{Y}=0$")
    ax.scatter(X_s_1[y_s_1 == 0][:, 0], X_s_1[y_s_1 == 0][:, 1],
            color='orange', marker='o', facecolors='none', s=30, label=r"$Y=0, \bar{Y}=1$")

    # Y = 1
    ax.scatter(X_s_0[y_s_0 == 1][:, 0], X_s_0[y_s_0 == 1][:, 1],
            color='green', marker='x', s=32, linewidth=1.4, label=r"$Y=1, \bar{Y}=0$")
    ax.scatter(X_s_1[y_s_1 == 1][:, 0], X_s_1[y_s_1 == 1][:, 1],
            color='lightgreen', marker='o', facecolors='none', s=30, label=r"$Y=1, \bar{Y}=1$")



    # --- Legend ---
    handles, labels = ax.get_legend_handles_labels()
    handles_p,   _ = contour_p.legend_elements()
    handles_f,   _ = contour_f.legend_elements()
    handles_all, _ = contour_all.legend_elements()

    if legend_flag:
        new_labels  = [f'Disc. Factor: {disc_factor:.2f}', rf'$\lambda$: {a:.1f}'] + labels + ['Perf. Reg','Fair. Reg','Fair-aware Reg']
        new_handles = [plt.Line2D([0], [0], color='black', linewidth=0),
                       plt.Line2D([0], [0], color='black', linewidth=0)] \
                      + handles + handles_p + handles_f + handles_all
        ax.legend(new_handles, new_labels, loc=2, fontsize=10, frameon=True)
    else:
        ax.legend(handles, labels, loc=2, fontsize=10, frameon=True)

    # --- Axes cosmetics ---
    ax.set_title(f"Synthetic Data – disc_factor={disc_factor:.2f}")
    ax.set_xlim(-15, 10)
    ax.set_ylim(-10, 15)
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)  # remove x-axis ticks
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)  # remove y-axis ticks
    
def subplot_reg_without_y_pred(ax, 
            Y_pred_p, Y_pred_f, 
            xx, yy, 
            X_s_0, y_s_0, X_s_1, y_s_1, 
            disc_factor= None, lmd= None, legend_flag=True,):
    """
    Draw decision boundaries and scatter points into a given Axes (ax).
    """

    # --- Contours ---
    contour_p   = ax.contour(xx, yy, Y_pred_p,   levels=[0.5], linestyles='--', linewidths=1.2, colors='green')
    contour_f   = ax.contour(xx, yy, Y_pred_f,   levels=[0.5], linestyles='--', linewidths=1.2, colors='orange')

    # --- Data points ---
    # Y = 0
    ax.scatter(X_s_0[y_s_0 == 0][:, 0], X_s_0[y_s_0 == 0][:, 1],
            color='red', marker='x', s=32, linewidth=1.4, label=r"$Y=0, \bar{Y}=0$")
    ax.scatter(X_s_1[y_s_1 == 0][:, 0], X_s_1[y_s_1 == 0][:, 1],
            color='orange', marker='o', facecolors='none', s=30, label=r"$Y=0, \bar{Y}=1$")

    # Y = 1
    ax.scatter(X_s_0[y_s_0 == 1][:, 0], X_s_0[y_s_0 == 1][:, 1],
            color='green', marker='x', s=32, linewidth=1.4, label=r"$Y=1, \bar{Y}=0$")
    ax.scatter(X_s_1[y_s_1 == 1][:, 0], X_s_1[y_s_1 == 1][:, 1],
            color='lightgreen', marker='o', facecolors='none', s=30, label=r"$Y=1, \bar{Y}=1$")

    # --- Legend ---
    handles, labels = ax.get_legend_handles_labels()
    handles_p,   _ = contour_p.legend_elements()
    handles_f,   _ = contour_f.legend_elements()

    if legend_flag:
        new_labels  =  labels + ['Perf. Reg','Fair. Reg']
        
        if lmd is not None:
            new_labels = new_labels + [rf'$\lambda$: {lmd:.1f}']
            
        if disc_factor is not None:
            new_labels = new_labels + [f'Disc. Factor: {disc_factor:.2f}']
        new_handles = handles + handles_p + handles_f
        ax.legend(new_handles, new_labels, loc=2, fontsize=13, frameon=True)
    
    

    # --- Axes cosmetics ---
    ax.set_xlim(-15, 10)
    ax.set_ylim(-10, 15)
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)


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

