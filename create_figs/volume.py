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
    
    pick_indices = [2, 4, 5, 6, 8]
    
    # Create 2 rows, share Y axis only across the same row
    fig, axes = plt.subplots(2, len(pick_indices), figsize=(14, 5), sharex=True, sharey='row', layout='compressed')

    for i, idx in enumerate(pick_indices):
        samples = C_preds_array[:, idx, :]
        mu = samples.mean(axis=0)
        
        k_values, normalized_volumes, cumulative_entropy = compute_metrics_vs_k(samples, mu, max_k=35)
        
        # --- Row 1: Normalized Volume ---
        ax_vol = axes[0, i]
        ax_vol.plot(k_values, normalized_volumes, '-o', color='tab:blue', markerfacecolor='none', linewidth=2, markersize=4)
        ax_vol.grid(True, alpha=0.3)
        ax_vol.set_yscale('log')
        if i == 0:
            ax_vol.set_ylabel("$\\bar{\\mathcal{V}}_k$")
            
        # --- Row 2: Cumulative Entropy ---
        ax_ent = axes[1, i]
        ax_ent.plot(k_values, cumulative_entropy, '-s', color='tab:green', markerfacecolor='none', linewidth=2, markersize=4)
        ax_ent.grid(True, alpha=0.3)
        ax_ent.set_xlabel("$k$")
        if i == 0:
            ax_ent.set_ylabel("$H_k$")

    # fig.suptitle("Inversion Ambiguity Metrics: Volume and Entropy vs. Complexity", fontsize=15)
    
    (BASE_DIR / 'figs').mkdir(exist_ok=True)
    out_path = BASE_DIR / 'figs/ambiguity_metrics.pdf'
    plt.savefig(out_path, bbox_inches='tight', dpi=1000)
    print(f"Saved plot to {out_path}")
    plt.show()

if __name__ == "__main__":
    main()