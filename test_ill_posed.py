import numpy as np
import matplotlib.pyplot as plt
from rnd_spectra.bumps import Bumps
from rnd_spectra.laplace import Laplace

class ControlledBumps(Bumps):
    def __init__(self, centers, widths, weights, hparams={}, random_seed=None):
        super().__init__(hparams, random_seed)
        self.centers = centers
        self.widths = widths
        self.weights = weights
        
    def random_spectrum(self):
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

centers1 = [2.0]
widths1 = [1.0]
weights1 = [1.0]

centers2 = [2.0, 8.0]
widths2 = [1.0, 0.5]
weights2 = [1.0, 0.5]

b1 = ControlledBumps(centers=centers1, widths=widths1, weights=weights1, random_seed=42)
l1 = Laplace(b1)
G1 = l1.evaluate_transformation()

b2 = ControlledBumps(centers=centers2, widths=widths2, weights=weights2, random_seed=42)
l2 = Laplace(b2)
G2 = l2.evaluate_transformation()

fig,axs = plt.subplots(1,2,figsize=(8,4))
axs[0].plot(l1.omega, l1.spectrum, label='Spectrum 1')
axs[0].plot(l2.omega, l2.spectrum, label='Spectrum 2')
axs[0].legend()
axs[1].plot(l1.tau, G1, label='G 1')
axs[1].plot(l2.tau, G2, label='G 2')
axs[1].legend()
plt.savefig('test_plot.png')
