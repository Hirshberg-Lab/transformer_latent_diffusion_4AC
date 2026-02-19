from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks",palette='tab10',rc={'axes.labelsize':14,'legend.fontsize': 10})
# import sys
# import os
# sys.path.append(os.path.dirname(os.getcwd())) 

from sklearn.decomposition import PCA
def compute_pcs_via_pca(samples, n_components=9):
    n_components = min(n_components, samples.shape[0])
    pca = PCA(n_components=n_components)
    pca.fit(samples)
    return {
        'pcs':pca.components_, 
        'eigvals':pca.explained_variance_, 
        'explained_variance_ratio': pca.explained_variance_ratio_
        }

load_path = Path("spectra/data_with_diffusion.npz")
print(f"Loading data from {load_path}")
loaded = np.load(load_path)

C_preds_array = loaded["C_preds"]
G_test = torch.from_numpy(loaded["G_test"])
C_test = torch.from_numpy(loaded["C_test"])
C_preds_mean = C_preds_array.mean(0)
C_preds_std = C_preds_array.std(0)

load_path = Path("spectra/data_deterministic.npz")
loaded = np.load(load_path)
C_preds_det = loaded["C_preds_det"]

fig,axs = plt.subplots(3,5,figsize=(12,6),sharex=False,sharey=False, layout='compressed',gridspec_kw={'height_ratios': [1, 2, 2]})
palette = ['C' + str(x) for x in range(250000)]
n_ex=5
omega = np.linspace(0,50,1024)
tau = np.linspace(0,1,99)
pick_indices = [2,4,5,6,8]

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
alpha_dummy = np.linspace(-3,3,200)
cmap = plt.get_cmap('jet')
norm = Normalize(vmin=alpha_dummy.min(), vmax=alpha_dummy.max())


for j in range(0,n_ex):
    
    pc_0 = compute_pcs_via_pca(C_preds_array[:,pick_indices[j],:])['pcs'][0,:]
    [axs[2,j].plot(omega,C_preds_mean[pick_indices[j]]+alpha*pc_0, color=cmap(norm(alpha)), alpha=0.05) for alpha in alpha_dummy]
    axs[2,j].plot(omega,C_preds_mean[pick_indices[j]], color='k')
    axs[0,j].plot(tau,G_test[pick_indices[j]][0].view(99).cpu().detach().numpy(),'--', color='k')#palette[j])
    
    [axs[i,j].plot(omega,C_test[pick_indices[j]].view(1024).cpu().detach().numpy(),'--' ,color='k',label='Ground Truth') for i in range(1,3)]
    axs[1,j].plot(omega,C_preds_mean[pick_indices[j]], color='k',label='Mean Pred.')#palette[0])
    axs[1,j].plot(omega,C_preds_det[pick_indices[j]].flatten(),'--', color='tab:red',label='Regression')
    axs[1,j].fill_between(omega,C_preds_mean[pick_indices[j]]-C_preds_std[pick_indices[j]],C_preds_mean[pick_indices[j]]+C_preds_std[pick_indices[j]],color='tab:grey',alpha=0.6,label='STD Pred.')

    axs[1,j].set_xlabel(r'$\omega$',labelpad=-2)
    axs[2,j].set_xlabel(r'$\omega$',labelpad=-2)
    axs[0,j].set_xlabel(r'$\tau$',labelpad=-2)
axs[1,0].set_ylabel(r'C($\omega$)')
axs[0,0].set_ylabel(r'G($\tau$)')
axs[2,0].set_ylabel(r'C($\omega$)+$ \alpha v$')
axs[1,0].legend(loc='lower right')
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([1.0, 0.08, 0.01, 0.295])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label(r'$\alpha$', labelpad=-9)

plt.savefig('figs/three_rows_of_ac.pdf',bbox_inches='tight',dpi=1000)
# plt.show()