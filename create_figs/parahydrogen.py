from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks",palette='tab10',rc={'axes.labelsize':14,'legend.fontsize': 10})
# from volume_bar_chart import compute_metrics_vs_k
from pseudo_volume_bar_chart import compute_metrics_vs_k

omega = np.linspace(0,50,1024)
kB = 3.1668114E-6  # in Hartree/K
c = 2.99792458E10  # speed of light in cm/s
s_in_t_au = 2.4188843265857E-17  # 1 time a.u. in s
T = 14.0  # in K
b = 1 / (kB * T)

# rabani output:
import pandas as pd
rabani_data = pd.read_csv("data/Rabani_D_w_14K.csv", sep=",", header=None)
rabani_data_w, rabani_data_D = rabani_data[0].values,rabani_data[1].values
rabani_data_w *= ( 2 * np.pi * c * s_in_t_au * b ) # now omega unitless

rabani_D_interp = np.interp(omega, rabani_data_w, rabani_data_D)
rabani_D_interp/= np.trapz(y=rabani_D_interp * (1-np.exp(-omega) + 1e-30)/(omega + 1e-30),x=omega)



mass=2.01588 # gram/mol
hbar = 1.054571817 * 1E-34 # J * sec = kg * m^2 /sec
hbar *= 1E+20 # kg * angstrom^2 /sec
hbar *= 1E-12 # kg * angstrom^2 /ps
mass /= (1000 * 6.02214076 * 1E+23) # kg
renorm_factor = hbar*3*np.pi/mass


load_path = BASE_DIR / "spectra/diffusion_rabani_pred.npz"
print(f"Loading data from {load_path}")
loaded = np.load(load_path)

C_preds_array = loaded["C_PIMD_input_preds"]
C_preds_mean_rabani = C_preds_array.mean(0)
C_preds_std_rabani = C_preds_array.std(0)

fig,ax = plt.subplots(figsize=(4,3),layout='compressed')
ax.plot(omega,renorm_factor*C_preds_mean_rabani[0],label='Diffusion Model Pred.',color='tab:blue')
ax.fill_between(omega,renorm_factor*(C_preds_mean_rabani[0]-C_preds_std_rabani[0]),renorm_factor*(C_preds_mean_rabani[0]+C_preds_std_rabani[0]),color='tab:blue',alpha=0.5)
print(renorm_factor*C_preds_mean_rabani[0][0]/6)
print(renorm_factor*C_preds_std_rabani[0][0]/6)
ax.plot(rabani_data_w,rabani_data_D*6,'--',color='orange',label='Max Entropy Pred.')
print(rabani_data_D[0])
ax.legend()
ax.set_xlabel(r'$\beta\hbar\omega$')
ax.set_ylabel(r'C($\omega$) [Å$^2$/ps]')

plt.savefig("figs/parahydrogen.pdf",bbox_inches='tight',dpi=1000)



################################################
samples = C_preds_array[:,0,:]
samples/=np.mean(np.linalg.norm(samples, axis=1))
mu = samples.mean(0)
k_values, normalized_volumes = compute_metrics_vs_k(samples, mu, max_k=999,c=0.34589846730232243)

print(f"Normalized Geometric Volume: {normalized_volumes[-1]}")
# print(f"Cumulative Posterior Entropy: {cumulative_entropy[-1]}")

#####################################################

# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable
# alpha_dummy = np.linspace(-3,3,200)
# cmap = plt.get_cmap('jet')
# norm = Normalize(vmin=alpha_dummy.min(), vmax=alpha_dummy.max())

# from comparison import compute_pcs_via_pca

# pc_0 = compute_pcs_via_pca(samples)['pcs'][0,:]

# fig,ax = plt.subplots(figsize=(4,3),layout='compressed')
# ax.plot(omega,renorm_factor*C_preds_mean_rabani[0],label='Diffusion Model Pred.',color='tab:blue')
# [ax.plot(omega,renorm_factor*(C_preds_mean_rabani[0]+alpha*pc_0), color=cmap(norm(alpha)), alpha=0.05) for alpha in alpha_dummy]
# ax.plot(rabani_data_w,rabani_data_D*6,'--',color='orange',label='Max Entropy Pred.')
# ax.legend()
# ax.set_xlabel(r'$\beta\hbar\omega$')
# ax.set_ylabel(r'C($\omega$) [Å$^2$/ps]')
# plt.savefig("figs/parahydrogen_pca.pdf",bbox_inches='tight',dpi=1000)