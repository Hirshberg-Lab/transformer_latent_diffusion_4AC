from scipy.stats import alpha
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

fig,axs = plt.subplots(1,4,figsize=(9,2.4),sharex=False,sharey=False, layout='compressed')
palette = ['C' + str(x) for x in range(250000)]
n_ex=4
omega = np.linspace(0,50,1024)
tau = np.linspace(0,1,99)
pick_indices = [2,4,5,6,8]

for channel in range(n_ex):
    for j in range(len(pick_indices)):
        axs[channel].plot(tau,G_test[pick_indices[j]][channel].view(99).cpu().detach().numpy(), linewidth=2.2,color=palette[j],alpha=0.8)
    
    axs[channel].set_xlabel(r'$\tau$',labelpad=-2)

axs[0].set_ylabel(r'$G(\tau)$')
axs[1].set_ylabel(r'log$(G(\tau))$')
axs[2].set_ylabel(r'$(G(\tau)-\mu(\tau))/\sigma(\tau)$')
axs[3].set_ylabel(r'$F_{\tau}(G(\tau))$')

plt.savefig(BASE_DIR / 'figs/four_channel_example.pdf',bbox_inches='tight',dpi=1000)
# plt.show()