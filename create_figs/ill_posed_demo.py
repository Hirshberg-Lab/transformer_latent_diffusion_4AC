from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks",palette='tab10',rc={'axes.labelsize':14,'legend.fontsize': 10})

from rnd_spectra.bumps import Bumps
from rnd_spectra.laplace import Laplace

class ControlledBumps(Bumps):
    def __init__(self, centers, widths, weights, hparams: dict = None, random_seed: int = None):
        if hparams is None:
            hparams = {}
        super().__init__(hparams, random_seed)
        self.centers = centers
        self.widths = widths
        self.weights = weights
        
    def random_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        spectrum = np.zeros(self.hparams['num_spectrum_points'])
        omega = np.linspace(*self.hparams['omega_domain'], self.hparams['num_spectrum_points'])
        for c, w, weight in zip(self.centers, self.widths, self.weights):
            params = self._generate_random_params()
            bump_current = self.bump_single(params)
            bump_current_shifted = np.interp(omega,
                                             w * np.linspace(-1.0, 1.0, self.hparams['num_spectrum_points']) + c,
                                             bump_current)
            spectrum += weight * bump_current_shifted
        spectrum = spectrum / np.max(spectrum)
        return spectrum, omega


fig,axs = plt.subplots(1,2,figsize=(7,2.7), layout='compressed')

# Left panel: 2 examples of spectra
centers1 = [3.55]
widths1 = [11.0]
weights1 = [2.0]

b1 = ControlledBumps(centers=centers1, widths=widths1, weights=weights1, random_seed=2)
l1 = Laplace(b1)
G1 = l1.evaluate_transformation()

centers2 = [2.0, 6.0]
widths2 = [4.0, 3.9]
weights2 = [1.0, 0.65]

b2 = ControlledBumps(centers=centers2, widths=widths2, weights=weights2, random_seed=42)
l2 = Laplace(b2)
G2 = l2.evaluate_transformation()

axs[0].plot(l1.omega, l1.spectrum, label=r'C$_1$',lw=3)
axs[0].plot(l2.omega, l2.spectrum, label=r'C$_2$', linestyle='--',lw=3)
axs[0].legend()

# Right panel: Their coresponding G(tau)s
axs[1].plot(l1.tau, G1, label=r'G$_1$',lw=3)
axs[1].plot(l2.tau, G2, label=r'G$_2$', linestyle='--',lw=3)
axs[1].legend()
axs[0].set_xlabel(r'$\omega$')
axs[0].set_ylabel(r'C($\omega$)')

axs[1].set_xlabel(r'$\tau$')
axs[1].set_ylabel(r'G($\tau$)')

plt.savefig("figs/ill_posed_demo.pdf",bbox_inches='tight',dpi=1000)