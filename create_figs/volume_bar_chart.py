import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Use design defaults
sns.set_theme(style="ticks", palette='tab10', rc={'axes.labelsize': 14, 'legend.fontsize': 10})

def compute_metrics_vs_k(samples, mu, max_k=15, confidence_level=0.90, epsilon=1e-16):
    """
    Calculates the Normalized Geometric Volume (Option 1) and 
    Cumulative Posterior Entropy (Option 2) as functions of k.
    """
    # 1. Setup
    lower_percentile = (1.0 - confidence_level) / 2.0 * 100
    upper_percentile = (confidence_level + (1.0 - confidence_level) / 2.0) * 100
    
    centered_samples = samples - mu
    max_k = min(max_k, samples.shape[0] - 1, samples.shape[1])
    
    # Fit PCA ONCE
    pca = PCA(n_components=max_k)
    projections = pca.fit_transform(centered_samples)
    
    # ==========================================
    # Option 1: Normalized Relative Geometric Volume
    # ==========================================
    L_p = np.percentile(projections, lower_percentile, axis=0)
    U_p = np.percentile(projections, upper_percentile, axis=0)
    widths = U_p - L_p
    
    k_values = np.arange(1, max_k + 1)
    geometric_volumes = np.zeros(max_k)
    
    for k in range(1, max_k + 1):
        log_mean = np.mean(np.log(widths[:k] + epsilon))
        v_k = np.exp(log_mean) - epsilon
        geometric_volumes[k-1] = np.maximum(0, v_k) 
        
    # Normalize by the average of the samples' L2 norms.
    mean_l2 = np.mean(np.linalg.norm(samples, axis=1))
    if mean_l2 > 0:
        normalized_volumes = geometric_volumes / mean_l2
    else:
        normalized_volumes = geometric_volumes
        
    # ==========================================
    # Option 2: Cumulative Posterior Entropy
    # ==========================================
    # Calculate global variance to properly normalize the probabilities
    total_variance = np.sum(np.var(centered_samples, axis=0))
    
    # Extract eigenvalues (variances) of the PCA components
    eigenvalues = pca.explained_variance_
    
    # Convert to probabilities
    p = eigenvalues / total_variance
    p_safe = np.maximum(p, epsilon) # Prevent log(0)
    
    # Calculate Shannon entropy terms and accumulate them over k
    entropy_terms = -p * np.log(p_safe) / np.log(samples.shape[0])
    cumulative_entropy = np.cumsum(entropy_terms)
    
    return k_values, normalized_volumes, cumulative_entropy


def main():
    load_path = BASE_DIR / "spectra/data_with_diffusion.npz"
            
    print(f"Loading data from {load_path}")
    try:
        loaded = np.load(load_path)
    except FileNotFoundError:
        print(f"Error: File {load_path} not found.")
        return

    C_preds_array = loaded["C_preds"]
    C_test = loaded["C_test"]
    if hasattr(C_test, 'numpy'):
        C_test = C_test.numpy()
    
    pick_indices = [2, 4, 5, 6, 8]
    target_k_values = [999]
    
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(10, 4.5))
    gs_main = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    
    # Top row with large panels (metrics)
    gs_metrics = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0])
    # Bottom row with small wspace (spectra)
    gs_spectra = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs_main[1], wspace=0.1)
    
    axes = {}
    for i in range(5):
        axes[f's{i}'] = fig.add_subplot(gs_spectra[0, i])
    axes['v'] = fig.add_subplot(gs_metrics[0, 0])
    axes['h'] = fig.add_subplot(gs_metrics[0, 1])
    
    num_k = len(target_k_values)
    num_indices = len(pick_indices)
    
    v_k_data = np.zeros((num_k, num_indices))
    h_k_data = np.zeros((num_k, num_indices))
    
    colors = sns.color_palette('tab10', n_colors=num_indices)
    
    ax_vol = axes['v']
    ax_ent = axes['h']
    
    omega = np.linspace(0, 50, 1024)
    
    for i, idx in enumerate(pick_indices):
        samples = C_preds_array[:, idx, :]
        mu = samples.mean(axis=0)
        
        # Plot bottom panel spectra
        ax_s = axes[f's{i}']
        ax_s.set_xticks([])
        ax_s.set_yticks([])
        # if i > 0:
            # ax_s.sharey(axes['s0'])
            # ax_s.tick_params(labelleft=False)
            
        ground_truth = C_test[idx].flatten()
        ax_s.plot(omega, ground_truth, '--', color='k', label='GT' if i==0 else None)
        ax_s.plot(omega, mu, '-', color=colors[i], linewidth=2, label='Mean' if i==0 else None)
        # [ax_s.plot(omega, samples[l], '-', color=colors[i], linewidth=0.5, alpha=0.02) for l in range(len(samples[:,0]))]
        
        if i == 0:
            ax_s.set_ylabel(r'C($\omega$)')
            # ax_s.legend(fontsize=8)
        ax_s.set_xlabel(r'$\omega$', labelpad=-1.)
        # ax_s.set_yticks([])
        
        k_values, normalized_volumes, cumulative_entropy = compute_metrics_vs_k(samples, mu, max_k=max(target_k_values))
        
        for j, k in enumerate(target_k_values):
            if k in k_values:
                k_idx = np.where(k_values == k)[0][0]
                v_k_data[j, i] = normalized_volumes[k_idx]
                h_k_data[j, i] = cumulative_entropy[k_idx]
            else:
                v_k_data[j, i] = np.nan
                h_k_data[j, i] = np.nan
                print(f"Warning: k={k} not found for index {idx}. Max k is {k_values[-1] if len(k_values) > 0 else 0}")
    
    # Plotting top panels (bar charts)
    x = np.arange(num_k)  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0
    
    for i, idx in enumerate(pick_indices):
        offset = width * multiplier
        ax_vol.bar(x + offset, v_k_data[:, i], width, label=f'index={idx}', color=colors[i])
        ax_ent.bar(x + offset, h_k_data[:, i], width, label=f'index={idx}', color=colors[i])
        multiplier += 1

    # Customize axes
    tick_locations = x + width * (num_indices - 1) / 2
    
    ax_vol.set_ylabel("$\\bar{\\mathcal{V}}$",labelpad=-1.)
    ax_vol.grid(True, alpha=0.3, axis='y')
    ax_vol.set_xticks([])

    ax_ent.set_ylabel("$H$",labelpad=-1.)
    ax_ent.grid(True, alpha=0.3, axis='y')
    ax_ent.set_xticks([])

    # fig.suptitle("Inversion Ambiguity Metrics per k", fontsize=16)
    # fig.tight_layout()
    
    (BASE_DIR / 'figs').mkdir(exist_ok=True)
    out_path = BASE_DIR / 'figs/ambiguity_metrics_bar.pdf'
    plt.savefig(out_path, bbox_inches='tight', dpi=1000)
    print(f"Saved plot to {out_path}")
    
if __name__ == "__main__":
    main()
