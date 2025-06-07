import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


class Bumps:

	def __init__(self, hparams: dict = {}, random_seed: int = None):
		"""
		Initialize a Bumps object for generating random power spectra.
		
		Args:
			hparams (dict): Optional dictionary to override default hyperparameters
			random_seed (int): Optional seed for random number generation
		"""
		self.hparams = {
        	'num_spectrum_points': 1024,  # The number of points in the spectrum.
        	'num_control_points_range': (5, 50),  # The range of the number of control points for the random warping of the basic bump function.
        	'minimal_fraction_of_maximal_control_point_jump': 0.05,  # The minimal fraction of the maximal control point jump; see _sample_control_points.
        	'prob_inertia_range': (0.0, 0.5),  # Controls the random warping of the basic bump function; see _sample_control_points.
        	'relative_inertia_range': (0.1, 0.4),  # Controls the random warping of the basic bump function; see _sample_control_points.
        	'fraction_points_for_smoothing': 0.05,  # The fraction of points in the kernel which smoothes the individual random bump functions.
        	'omega_domain': (0, 10.0),  # The domain of the frequency of the random power spectrum.
        	'num_bumps_range': (1, 6),  # The range of the number of bumps in the random power spectrum.
        	'bump_weights_range': (0.2, 1.0),  # The range of the bump weights.
        	'bump_widths_fraction_range': (0.03, 0.15),  # The range of the bump widths, as a fraction of the domain of the random power spectrum.
        	'bump_centers_fraction_range': (0.0, 0.8)   # The range of the bump centers, as a fraction of the domain of the random power spectrum.
		}
		self.hparams = self.hparams | hparams
		self.rng = np.random.default_rng(random_seed)

	def _bump_basic(self, omega: np.ndarray) -> np.ndarray:
		"""
		Generate a basic bump function.
		This is a Gaussian which is nearly zero at the edges, and has a maximum value of 1 in the center.
		
		Args:
			omega (np.ndarray): The frequency domain
			
		Returns:
			spectrum (np.ndarray): The basic bump function
		"""
		width = 0.1  # Chosen so that the Gaussian has died off at 0 and 1.
		center = 0.5
		spectrum = np.exp(-(omega - center)**2 / (2 * width**2))
		return spectrum

	def _sample_control_points(self, params: dict) -> np.ndarray:
		"""
		Sample control points randomly for a bump function.
		The control points form the basis of the random warping of the basic bump function, which is a Gaussian.
		
		Args:
			params (dict): Dictionary containing parameters for the bump function
			
		Returns:
			control_points (np.ndarray): Array of control points
		"""
		control_point_jump_range = (self.hparams['minimal_fraction_of_maximal_control_point_jump'], 1.0)
		control_points = np.zeros(params['num_control_points']-1)
		control_points[0] = self.rng.uniform(*control_point_jump_range)
		for i in range(1, params['num_control_points']-1):
			r = self.rng.uniform(0.0, 1.0)
			if r < params['prob_inertia']:
				control_points[i] = self.rng.uniform((1-params['relative_inertia'])*control_points[i-1],
													 (1+params['relative_inertia'])*control_points[i-1])
			else:
				control_points[i] = self.rng.uniform(*control_point_jump_range)
		control_points = np.append([0.0], np.cumsum(control_points))
		control_points = control_points / control_points[-1]
		return control_points

	def _recenter(self, spectrum: np.ndarray) -> np.ndarray:
		"""
		Re-center the spectrum to have a maximum value at the center.
		
		Args:
			spectrum (np.ndarray): The spectrum to recenter
			
		Returns:
			spectrum_recentered (np.ndarray): The recentered spectrum with maximum at the center
		"""
		center = np.argmax(spectrum)
		center_actual = np.round(spectrum.size / 2.0).astype(int)
		spectrum_recentered = np.zeros(spectrum.shape)
		if center < center_actual:
			shift = center_actual - center
			spectrum_recentered[shift:] = spectrum[:-shift]
		elif center > center_actual:
			shift = center - center_actual
			spectrum_recentered[:-shift] = spectrum[shift:]
		else:
			spectrum_recentered = spectrum
		return spectrum_recentered

	def bump_single(self, params: dict) -> np.ndarray:
		"""
		Generate a single bump function, which is a warped Gaussian.
		The domain of the bump function is omega in [0, 1].
		This domain is shifted when combining bumps together to create the spectrum.
		
		Args:
			params (dict): Dictionary containing parameters for the bump function
			
		Returns:
			spectrum (np.ndarray): The single bump function
		"""
		omega_control_points = self._sample_control_points(params)
		omega_warped = np.interp(np.linspace(0.0, 1.0, self.hparams['num_spectrum_points']),
								 np.linspace(0.0, 1.0, params['num_control_points']),
								 omega_control_points)
		spectrum = self._bump_basic(omega_warped)
		smoother_size = np.round(self.hparams['num_spectrum_points'] * self.hparams['fraction_points_for_smoothing']).astype(int)
		smoother = self._bump_basic(np.linspace(0.0, 1.0, smoother_size))
		smoother = smoother / np.sum(smoother)
		spectrum = np.convolve(spectrum, smoother, mode='same')
		spectrum = spectrum / np.max(spectrum)
		spectrum = self._recenter(spectrum)
		return spectrum

	def _generate_random_params(self) -> dict:
		"""
		Generate random parameters for which generate a bump function.
		These parameters form the basis of the random warping of the basic bump function, which is a Gaussian.
		
		Returns:
			params (dict): Dictionary containing the generated parameters
		"""
		params = {
			'num_control_points': self.rng.integers(*self.hparams['num_control_points_range']),
			'prob_inertia': self.rng.uniform(*self.hparams['prob_inertia_range']),
			'relative_inertia': self.rng.uniform(*self.hparams['relative_inertia_range'])
		}
		return params

	def random_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
		"""
		Generate a random power spectrum.
		This is a sum of bumps, which are warped Gaussians.

		Returns:
			spectrum (np.ndarray): The generated power spectrum
			omega (np.ndarray): The corresponding frequency domain
		"""
		num_bumps = self.rng.integers(*self.hparams['num_bumps_range'])
		bump_weights = self.rng.uniform(*self.hparams['bump_weights_range'], num_bumps)
		bump_widths = self.rng.uniform(*self.hparams['bump_widths_fraction_range'], num_bumps) * self.hparams['omega_domain'][1]
		bump_centers = self.rng.uniform(*self.hparams['bump_centers_fraction_range'], num_bumps) * self.hparams['omega_domain'][1]
		spectrum = np.zeros(self.hparams['num_spectrum_points'])
		omega = np.linspace(*self.hparams['omega_domain'], self.hparams['num_spectrum_points'])
		for i in range(num_bumps):
			params = self._generate_random_params()
			bump_current = self.bump_single(params)
			bump_current_shifted = np.interp(omega,
            								 bump_widths[i] * np.linspace(-1.0, 1.0, self.hparams['num_spectrum_points']) + bump_centers[i],
											 bump_current)
			spectrum += bump_weights[i] * bump_current_shifted
		spectrum = spectrum / np.max(spectrum)
		return spectrum, omega


class BumpTest:

	def __init__(self, bumps: Bumps):
		self.bumps = bumps

	def plot_bump_single(self) -> None:
		params = self.bumps._generate_random_params()
		bump = self.bumps.bump_single(params)
		domain = np.linspace(0.0, 1.0, self.bumps.hparams['num_spectrum_points'])
		plt.plot(domain, bump)
		plt.show()
		return

	def plot_random_spectrum(self) -> None:
		spectrum, omega = self.bumps.random_spectrum()
		plt.plot(omega, spectrum)
		plt.show()
		return