import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Use design defaults from comparison2.py
sns.set_theme(style="ticks", palette='tab10', rc={'axes.labelsize': 14, 'legend.fontsize': 10})

def main():
    load_path = BASE_DIR / "spectra/data_with_diffusion.npz"
    print(f"Loading data from {load_path}")
    loaded = np.load(load_path)

    C_preds_array = loaded["C_preds"]
    C_test = torch.from_numpy(loaded["C_test"])
    
    n_rows = 5
    n_cols = 6
    # Adjusted figsize for better visual spacing of 15 columns
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 9), sharex=True, sharey='row', layout='compressed')
    
    omega = np.linspace(0, 50, 1024)
    # The columns from the 2nd row in comparison2.py correspond to these indices
    pick_indices = [2, 4, 5, 6, 8]

    for j in range(n_rows):
        idx = pick_indices[j]
        
        # Extract ground truth similar to uncertainty_analysis.py
        if hasattr(C_test, 'numpy'):
            ground_truth = C_test[idx].numpy().flatten()
        else:
            ground_truth = C_test[idx].flatten()
            
        samples = C_preds_array[:, idx, :]
        mu = samples.mean(axis=0)
        
        # Perform PCA
        pca = PCA(n_components=min(n_cols, samples.shape[0]))
        pca.fit(samples)
        components = pca.components_
        
        centered_gt = ground_truth - mu
        coeffs_p = components @ centered_gt
        
        for k_col in range(n_cols):
            # The k-th column uses k PCA components to reconstruct the spectra
            k = k_col   
            ax = axs[j, k_col]
            
            # Reconstruction logic from plot_reconstruction_evolution
            delta = components[:k].T @ coeffs_p[:k]
            recon = mu + delta
                
            # Plot elements
            # Ground truth line styling matches comparison2.py
            ax.plot(omega, ground_truth, '--', color='k', label='Ground Truth' if j == 0 and k_col == n_cols - 1 else None)
            
            # Reconstruction line styling based on plot_reconstruction_evolution
            ax.plot(omega, recon, color='tab:blue', linewidth=2, label='PCA Recon.' if j == 0 and k_col == n_cols - 1 else None)
            
            # Calculate and display reconstruction error
            error = np.linalg.norm(ground_truth - recon) / np.linalg.norm(ground_truth)
            
            # Cumulative explained variance up to k components
            if k == 0:
                ev = 0.0
            else:
                ev = np.sum(pca.explained_variance_ratio_[:k])
            
            # Extract the k-th component scalar and formulate text string
            if k == 0:
                text_str = f'$r_{{{k}}}$={error:.0%}\nEV={ev:.0%}'
            else:
                c_val = coeffs_p[k-1]
                text_str = f'$r_{{{k}}}$={error:.0%}\n$\\alpha_{{{k}}}$={c_val:.2f}\nEV={ev:.0%}'

            ax.text(0.95, 0.5, text_str, 
                    transform=ax.transAxes, 
                    ha='right', va='center', 
                    fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
            # Apply design defaults for limits/labels
            if j == n_rows - 1:
                ax.set_xlabel(r'$\omega$', labelpad=-2)
            if k_col == 0:
                ax.set_ylabel(r'C($\omega$)')
                
            if j == 0:
                ax.set_title(f'$k={k}$')

    # Add legend to the top-rightmost plot
    axs[0, n_cols - 1].legend(loc='upper right')

    (BASE_DIR / 'figs').mkdir(exist_ok=True)
    out_path = BASE_DIR / 'figs/visualize_convergence.pdf'
    plt.savefig(out_path, bbox_inches='tight', dpi=1000)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
