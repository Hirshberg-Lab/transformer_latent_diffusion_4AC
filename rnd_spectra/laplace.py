import numpy as np
import matplotlib.pyplot as plt
from rnd_spectra.bumps import Bumps

class Laplace:

    def __init__(self, bumps: Bumps, n_tau_grid: int =99) -> None:
        self.bumps = bumps
        self.omega = bumps.random_spectrum()[1]
        self.tau = np.linspace(0,1,n_tau_grid)
        self.kernel = np.exp( -self.omega * self.tau[:,None] ) \
            + np.exp( self.omega * ( self.tau[:,None] -1 ) ) 
        self.eps = 1e-30 # for numerical stability
    def _create_spectrum(self) -> None:
        spectrum = self.bumps.random_spectrum()[0]
        integrand = spectrum * ( 1 - np.exp(-self.omega) + self.eps ) / ( self.omega + self.eps )
        A = np.trapz(y=integrand,x=self.omega)
        self.spectrum = spectrum/A
    def evaluate_transformation(self) -> np.ndarray:
        self._create_spectrum()
        integrand = self.kernel * self.spectrum /2./np.pi
        return np.trapz(y=integrand,x=self.omega)


class LaplaceTest:

    def __init__(self, bumps: Bumps) -> None:
        self.laplace = Laplace(bumps)
        self.tau = self.laplace.tau
        self.omega = self.laplace.omega
    
    def plot_spectrum_and_itcf(self) -> None:
        fig, ax = plt.subplots(1,2,figsize=(8,3))
        ax[1].plot(self.tau, self.laplace.evaluate_transformation())
        ax[0].plot(self.omega, self.laplace.spectrum)
        ax[0].set_xlabel(r'$\omega$'), ax[0].set_ylabel(r'C($\omega$)')
        ax[1].set_xlabel(r'$\tau$'), ax[1].set_ylabel(r'G($\tau$)')
        plt.tight_layout()
        plt.show()
        return
