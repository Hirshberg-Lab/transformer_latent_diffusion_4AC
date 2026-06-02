from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks",palette='tab10',rc={'axes.labelsize':14,'legend.fontsize': 10})

load_path = BASE_DIR / "spectra/data_with_diffusion.npz"
print(f"Loading data from {load_path}")
loaded = np.load(load_path)

C_preds_array = loaded["C_preds"]
G_test = torch.from_numpy(loaded["G_test"])
C_test = torch.from_numpy(loaded["C_test"])
C_preds_mean = C_preds_array.mean(0)
C_preds_std = C_preds_array.std(0)

det_files = list((BASE_DIR / "spectra").glob("data_deterministic_*.npz"))
C_preds_det_list = []
for f in det_files:
    loaded = np.load(f)
    C_preds_det_list.append(loaded["C_preds_det"])
C_preds_det_array = np.array(C_preds_det_list)
C_preds_det_mean = C_preds_det_array.mean(0)
C_preds_det_std = C_preds_det_array.std(0)

fig,axs = plt.subplots(2,5,figsize=(13,4),sharex=False,sharey=False, layout='compressed',gridspec_kw={'height_ratios': [1, 2]})
palette = ['C' + str(x) for x in range(250000)]
n_ex=5
omega = np.linspace(0,50,1024)
tau = np.linspace(0,1,99)
pick_indices = [2,4,5,6,8]

for j in range(0,n_ex):
    
    # Row 0
    axs[0,j].plot(tau,G_test[pick_indices[j]][0].view(99).cpu().detach().numpy(),'--', color='k')
    
    # Row 1
    
    # [axs[2,j].plot(omega,C_preds_array[i,pick_indices[j]], 
    # color='tab:purple', alpha=0.5,linewidth=0.8) for i in range(20)]
    # axs[2,j].plot(omega,C_preds_mean[pick_indices[j]], color='tab:blue',linewidth=2.9)
    axs[1,j].plot(omega,C_test[pick_indices[j]].view(1024).cpu().detach().numpy(),'--',linewidth=2.5 ,color='k',label='Ground Truth')
    # axs[1,j].plot(0,0,color='tab:purple', alpha=0.5,linewidth=1.5, label='Diffusion Model Predictions')
    axs[1,j].plot(omega,C_preds_mean[pick_indices[j]], color='tab:blue',linewidth=2.9,label='Mean Pred.')
    
    # axs[1,j].fill_between(omega, (C_preds_det_mean[pick_indices[j]]-C_preds_det_std[pick_indices[j]]).flatten(), (C_preds_det_mean[pick_indices[j]]+C_preds_det_std[pick_indices[j]]).flatten(), color='tab:orange', alpha=0.3)
    axs[1,j].fill_between(omega,C_preds_mean[pick_indices[j]]-C_preds_std[pick_indices[j]],C_preds_mean[pick_indices[j]]+C_preds_std[pick_indices[j]],color='tab:blue',alpha=0.55,label='STD Pred.')
    
    axs[1,j].plot(omega,C_preds_det_mean[pick_indices[j]].flatten(),'--',linewidth=2, color='tab:orange',label='Regression Model Pred.',alpha=0.8)

    axs[1,j].set_xlabel(r'$\omega$',labelpad=-2)
    # axs[2,j].set_xlabel(r'$\omega$',labelpad=-2)
    axs[0,j].set_xlabel(r'$\tau$',labelpad=-2)

axs[1,0].set_ylabel(r'C($\omega$)')
# axs[2,0].set_ylabel(r'C($\omega$)')
axs[0,0].set_ylabel(r'G($\tau$)')
handles, labels = axs[1, 0].get_legend_handles_labels()
# axs[1,0].legend(loc='lower right')
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.09), ncol=5, fontsize=12,frameon=False)

# Create figs folder if it doesn't exist just in case
(BASE_DIR / 'figs').mkdir(exist_ok=True)
plt.savefig(BASE_DIR / 'figs/two_rows.pdf',bbox_inches='tight',dpi=1000)
