import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
import torch
import seaborn as sns
sns.set_theme(style="ticks",palette='tab10',rc={'axes.labelsize':14,'legend.fontsize': 10})


class SpectralAnalysis:
    def __init__(self, C_preds, C_test, index=0, frequencies=None):
        """
        :param C_preds: Shape (n_samples, n_examples, dim)
        :param C_test: Shape (n_examples, dim) - Can be Tensor or Numpy
        :param index: The specific example index to analyze (0 to n_examples-1)
        :param frequencies: Array of frequency values for x-axis (optional)
        """
        self.index = index
        # Select data for the specific example
        self.samples = C_preds[:, index, :]  # Shape (n_samples, dim)
        
        # Handle C_test being Tensor or Numpy
        if hasattr(C_test, 'numpy'):
             self.ground_truth = C_test[index].numpy().flatten() # Shape (dim,)
        else:
             self.ground_truth = C_test[index].flatten()
             
        self.dim = self.samples.shape[1]
        
        # Statistics for p_p (Correlated)
        self.mu = self.samples.mean(axis=0)
        self.cov_p = np.cov(self.samples, rowvar=False)
        
        # Statistics for p_u (Uncorrelated)
        self.sigma_u = self.samples.std(axis=0)
        self.cov_u = np.diag(self.sigma_u**2)
        
        # Frequencies for plotting
        if frequencies is None:
            self.frequencies = np.linspace(0, 1, self.dim)
        else:
            self.frequencies = frequencies


    def compute_nll(self, epsilon=1e-6):
        """
        Idea #2: Comparing NLL of the Ground Truth.
        Adds epsilon jitter to sigma_p for numerical stability.
        """
        # p_p NLL (Correlated)
        # Add jitter to ensure invertibility if samples < dim
        cov_p_robust = self.cov_p + epsilon * np.eye(self.dim)
        try:
            nll_p = -multivariate_normal.logpdf(self.ground_truth, mean=self.mu, cov=cov_p_robust)
        except np.linalg.LinAlgError:
            nll_p = np.inf

        # p_u NLL (Uncorrelated/Diagonal)
        cov_u_robust = self.cov_u + epsilon * np.eye(self.dim)
        nll_u = -multivariate_normal.logpdf(self.ground_truth, mean=self.mu, cov=cov_u_robust)

        return nll_p, nll_u

    def analyze_nll_breakdown(self, epsilon=1e-5):
        """
        Prints the Volume vs. Fit breakdown for both models.
        """
        # 1. Setup Covariances
        cov_p = self.cov_p + epsilon * np.eye(self.dim)
        cov_u = self.cov_u + epsilon * np.eye(self.dim)
        
        diff = self.ground_truth - self.mu

        # --- Analyze p_p ---
        sign, logdet_p = np.linalg.slogdet(cov_p)
        # Mahalanobis = diff.T * inv(Sigma) * diff
        inv_p = np.linalg.inv(cov_p) 
        mahal_p = diff @ inv_p @ diff.T
        nll_p = 0.5 * logdet_p + 0.5 * mahal_p

        # --- Analyze p_u ---
        sign, logdet_u = np.linalg.slogdet(cov_u)
        inv_u = np.linalg.inv(cov_u)
        mahal_u = diff @ inv_u @ diff.T
        nll_u = 0.5 * logdet_u + 0.5 * mahal_u

        print(f"--- NLL Breakdown (epsilon={epsilon}) ---")
        print(f"Model p_p (Correlated):")
        print(f"  Volume Penalty (LogDet): {0.5*logdet_p:.2f} (Usually Small/Negative)")
        print(f"  Fit Penalty (Mahalanobis): {0.5*mahal_p:.2f} (LIKELY HUGE -> The Problem)")
        print(f"  Total NLL: {nll_p:.2f}")
        
        print(f"\nModel p_u (Uncorrelated):")
        print(f"  Volume Penalty (LogDet): {0.5*logdet_u:.2f} (Usually Large)")
        print(f"  Fit Penalty (Mahalanobis): {0.5*mahal_u:.2f} (Moderate)")
        print(f"  Total NLL: {nll_u:.2f}")

        # --- Diagnostic: Compare Diagonals ---
        diag_p = np.diag(self.cov_p)
        diag_u = np.diag(self.cov_u)
        print(f"\n[Diagnostic] Variances (Diagonal of Covariance):")
        print(f"  Sum(Var(p_p)): {np.sum(diag_p):.2e}")
        print(f"  Sum(Var(p_u)): {np.sum(diag_u):.2e}")
        print(f"  Ratio (p_p/p_u): {np.sum(diag_p)/np.sum(diag_u):.2f} (Should be 1.0 if consistent)")

    def plot_parsimony(self, max_k=20, ax=None):
        """
        Idea #3: Reconstruction Error vs Model Complexity (k).
        """
        errors_p = []
        errors_u = []
        
        # Center the ground truth
        centered_gt = self.ground_truth - self.mu
        norm_centered_sq = np.linalg.norm(centered_gt)**2
        denominator = np.linalg.norm(self.ground_truth)
        
        # --- 1. p_p (PCA Basis) ---
        pca = PCA(n_components=min(max_k, self.samples.shape[0]))
        pca.fit(self.samples)
        components = pca.components_ # (k, dim)
        
        # Compute reconstruction error for k=0 to max_k
        # Coefficients: projection of GT onto components
        coeffs = components @ centered_gt
        
        # k=0 case: reconstruction is just the mean (which is 0 for centered_gt)
        # error is norm of centered_gt
        errors_p.append(np.linalg.norm(centered_gt) / denominator)

        for k in range(1, max_k + 1):
            if k > len(coeffs): break
            # Reconstruct using top k
            recon_vec = components[:k].T @ coeffs[:k]
            residual = np.linalg.norm(centered_gt - recon_vec)
            errors_p.append(residual / denominator)

        # --- 2. p_u (Frequency Basis) ---
        # "Best fit" in standard basis means picking frequencies with largest absolute amplitude
        
        # Alternatively: Sort by Variance (standard PCA logic applied to diagonal)
        # This is fairer to p_u as it picks the "noisiest" channels first
        # sorted_indices = np.argsort(self.sigma_u)[::-1]

        # Sort by MAGNITUDE of the signal (Strict "Best Fit"), not sigma
        # This gives p_u the best possible mathematical chance.
        coeffs_u_sorted_sq = np.sort(centered_gt**2)[::-1]
        
        captured_energy_u = np.cumsum(coeffs_u_sorted_sq)
        
        # k=0 case
        errors_u.append(np.linalg.norm(centered_gt) / denominator)

        for k in range(1, max_k + 1):
            if k > len(captured_energy_u): break
            residual_sq = np.maximum(0, norm_centered_sq - captured_energy_u[k-1])
            r_k = np.sqrt(residual_sq) / denominator
            errors_u.append(r_k)

        # Plot
        if ax is None:
            plt.figure(figsize=(8, 5),sharex=True)
            ax = plt.gca()
            
        ax.plot(range(0, len(errors_p)), errors_p, '-o',color='tab:blue', markerfacecolor='none', label=r'$p_p$ (Correlated)')
        ax.plot(range(0, len(errors_u)), errors_u, '--s',color='tab:red', markerfacecolor='none', label=r'$p_u$ (Uncorrelated)')
        ax.set_xlabel('$k$')
        # ax.set_ylabel(r'Normalized Error $||C - \hat{C}_k||/||C||$')
        # ax.set_title(f'Idea #3: Parsimony (Ex {self.index})')
        ax.grid(True, alpha=0.3)
        # ax.legend()
        if ax.get_figure() and not ax.get_figure().axes: # Check if standalone
             plt.show()

    def plot_spaghetti(self, n_plot_samples=30, ax=None):
        """
        Visualizing the Correlated Cloud vs Ground Truth.
        """
        if ax is None:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
        
        # Plot a subset of the actual generated samples
        # (These represent draws from p_p)
        indices = np.random.choice(self.samples.shape[0], min(n_plot_samples, self.samples.shape[0]), replace=False)
        subset = self.samples[indices]
        
        for s in subset:
            ax.plot(self.frequencies, s, color='blue', alpha=0.1, linewidth=1)
            
        # Plot Mean
        ax.plot(self.frequencies, self.mu, 'b--', linewidth=2, label=r'Prediction Mean $\mu$')
        
        # Plot Ground Truth
        ax.plot(self.frequencies, self.ground_truth, 'k-', linewidth=2.5, label='Ground Truth $C_{test}$')
        
        # Plot 2-sigma envelope (This represents p_u visually)
        upper = self.mu + 2 * self.sigma_u
        lower = self.mu - 2 * self.sigma_u
        ax.plot(self.frequencies, upper, 'r:', linewidth=1)
        ax.plot(self.frequencies, lower, 'r:', linewidth=1, label=r'$p_u$ 2$\sigma$ Bands')
        
        ax.set_title(f'Spaghetti Plot (Example {self.index})')
        ax.set_xlabel(r'Frequency $\omega$')
        ax.set_ylabel(r'$C(\omega)$')
        ax.legend()
        if ax.get_figure() and not ax.get_figure().axes:
             plt.show()

    def plot_correlation_heatmap(self, ax=None):
        """
        Visualizing the Covariance Structure.
        """
        # Calculate Correlation Matrix: Cov_ij / (std_i * std_j)
        std_outer = np.outer(self.sigma_u, self.sigma_u)
        # Avoid division by zero
        std_outer[std_outer == 0] = 1.0
        
        correlation = self.cov_p / std_outer
        self.correlation = correlation
        
        if ax is None:
            plt.figure(figsize=(7, 6))
            ax = plt.gca()

        im = ax.imshow(correlation, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
        # plt.colorbar(im, label='Correlation Coefficient')
        # Use extent to map indices to frequency values
        freq_min, freq_max = self.frequencies.min(), self.frequencies.max()
        im = ax.imshow(correlation, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower',
                       extent=[freq_min, freq_max, freq_min, freq_max])
        
        # plt.colorbar(im, label='Correlation Coefficient')
        # if ax.get_figure():
        #      ax.get_figure().colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Correlation Coefficient')
        
        # ax.set_title(f'Correlation Heatmap (Example {self.index})')
        ax.set_xlabel(r'$\omega$')
        # ax.set_ylabel(r'$\omega$')
        if ax.get_figure() and not ax.get_figure().axes:
             plt.show()

    def plot_reconstruction_evolution(self, k_steps=[0, 1, 3, 10]):
        """
        Visualizes C_hat_k approaching C_true as k increases.
        Compares p_p (PCA) vs p_u (Greedy Point Selection).
        """
        centered_gt = self.ground_truth - self.mu
        
        # --- Prepare p_p Components ---
        max_k = max(k_steps)
        pca = PCA(n_components=min(max_k, self.samples.shape[0]))
        pca.fit(self.samples)
        components = pca.components_ 
        coeffs_p = components @ centered_gt

        # --- Prepare p_u Indices ---
        # Sort indices by magnitude of deviation |C - mu|
        sorted_indices = np.argsort(np.abs(centered_gt))[::-1]

        # --- Plotting ---
        fig, axes = plt.subplots(2, len(k_steps), figsize=(4*len(k_steps), 8), sharex=True, sharey=True)
        
        # Row 1: p_p (Correlated / PCA)
        for i, k in enumerate(k_steps):
            ax = axes[0, i]
            
            # Construct C_hat_k
            if k == 0:
                recon = self.mu
            else:
                # mu + sum(alpha_i * v_i)
                delta = components[:k].T @ coeffs_p[:k]
                recon = self.mu + delta
            
            # Plot
            ax.plot(self.frequencies, self.ground_truth, 'k-', alpha=0.3, linewidth=3, label='Ground Truth')
            ax.plot(self.frequencies, recon, 'b-', linewidth=2, label=f'Reconstruction ($k={k}$)')
            
            # Formatting
            error = np.linalg.norm(self.ground_truth - recon) / np.linalg.norm(self.ground_truth)
            ax.set_title(f'$p_p$ (PCA)\n$k={k}$, Error={error:.2%}')
            if i == 0: ax.legend(loc='upper right', fontsize='small')

        # Row 2: p_u (Uncorrelated / Greedy)
        for i, k in enumerate(k_steps):
            ax = axes[1, i]
            
            # Construct C_hat_k
            recon = self.mu.copy()
            if k > 0:
                # The "Best" k points are set to Ground Truth, rest remain at Mean
                idx_to_fix = sorted_indices[:k]
                recon[idx_to_fix] = self.ground_truth[idx_to_fix]
            
            # Plot
            ax.plot(self.frequencies, self.ground_truth, 'k-', alpha=0.3, linewidth=3, label='Ground Truth')
            # Use a step plot or markers for p_u to highlight the "spikiness"
            ax.plot(self.frequencies, recon, 'r-', linewidth=1.5, label=f'Reconstruction ($k={k}$)')
            
            # Highlight the "fixed" points
            if k > 0:
                ax.plot(self.frequencies[sorted_indices[:k]], recon[sorted_indices[:k]], 'ro', markersize=4)

            # Formatting
            error = np.linalg.norm(self.ground_truth - recon) / np.linalg.norm(self.ground_truth)
            ax.set_title(f'$p_u$ (Greedy)\n$k={k}$, Error={error:.2%}')

        plt.suptitle(f"Visualizing Convergence: Physics ($p_p$) vs. Brute Force ($p_u$)", fontsize=16)
        plt.tight_layout()
        plt.show()

# ==========================================
# Execution
# ==========================================

if __name__ == "__main__":
    # 0. Load Data (Logic from comparison.py)
    load_path = BASE_DIR / "spectra/data_with_diffusion.npz"
    if not load_path.exists():
        print(f"Error: File {load_path} not found.")

    try:
        print(f"Loading data from {load_path}")
        loaded = np.load(load_path)
        C_preds_array = loaded["C_preds"]
        # G_test = torch.from_numpy(loaded["G_test"]) # Not needed for this analysis yet
        C_test = torch.from_numpy(loaded["C_test"])
    except FileNotFoundError:
        print(f"Could not load data. Please ensure {load_path} exists.")
        exit(1)

    # 1. Define frequencies (Logic from comparison.py)
    # comparison.py: omega = np.linspace(0,50,1024)
    omega = np.linspace(0,50,1024)

    # 2. Define which example to look at
    # comparison.py uses pick_indices = [2,4,5,6,8]
    indices_to_analyze = [2, 4, 5, 6, 8]
    
    # 3. Instantiate Analysis and Run Execution
    
    # NLL Comparison (Print only)
    print(f"--- Idea #2: NLL Comparison ---")
    for idx in indices_to_analyze:
        analysis = SpectralAnalysis(C_preds_array, C_test, index=idx, frequencies=omega)
        nll_p, nll_u = analysis.compute_nll()
        winner = 'p_p' if nll_p < nll_u else 'p_u'
        print(f"Ex {idx}: p_p={nll_p:.2f}, p_u={nll_u:.2f}, Winner={winner}")
        analysis.analyze_nll_breakdown()
    print("\n")

    # Idea #3: Parsimony Plot
    fig1, axes1 = plt.subplots(1, 5, figsize=(12, 3),sharex=True,sharey=True,layout='compressed')
    for i, idx in enumerate(indices_to_analyze):
        analysis = SpectralAnalysis(C_preds_array, C_test, index=idx, frequencies=omega)
        analysis.plot_parsimony(max_k=15, ax=axes1[i])
    axes1[0].set_ylabel(r'$r_k$')
    axes1[-1].legend()
    # fig1.suptitle("Idea #3: Parsimony Analysis")
    # fig1.tight_layout()
    # plt.savefig(BASE_DIR / 'figs/parsimony_analysis.pdf',bbox_inches='tight',dpi=1000)
    plt.show()

    # Spaghetti Plot
    fig2, axes2 = plt.subplots(1, 5, figsize=(24, 5))
    for i, idx in enumerate(indices_to_analyze):
        analysis = SpectralAnalysis(C_preds_array, C_test, index=idx, frequencies=omega)
        analysis.plot_spaghetti(ax=axes2[i])
    fig2.suptitle("Spaghetti Plots")
    fig2.tight_layout()
    plt.show()

    # Correlation Heatmap
    fig3, axes3 = plt.subplots(1, 5, figsize=(12, 3),layout='compressed',sharex=True,sharey=True)
    for i, idx in enumerate(indices_to_analyze):
        analysis = SpectralAnalysis(C_preds_array, C_test, index=idx, frequencies=omega)
        analysis.plot_correlation_heatmap(ax=axes3[i])
    im = axes3[-1].imshow(analysis.correlation, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower',
                       extent=[omega.min(), omega.max(), omega.min(), omega.max()])
    fig3.colorbar(im, pad=0.02, label='Correlation Coefficient')
    axes3[0].set_ylabel(r'$\omega$')
    # fig3.suptitle("Correlation Heatmaps")
    # fig3.tight_layout()
    # plt.savefig(BASE_DIR / 'figs/correlation_heatmap.pdf',bbox_inches='tight',dpi=1000)
    plt.show()

    # Idea #4: Reconstruction Evolution
    print(f"--- Idea #4: Reconstruction Evolution ---")
    for idx in indices_to_analyze:
        analysis = SpectralAnalysis(C_preds_array, C_test, index=idx, frequencies=omega)
        analysis.plot_reconstruction_evolution(k_steps=[0, 1, 3, 10, 20])
