from rnd_spectra.bumps import Bumps
from rnd_spectra.ontheflydataset import OnTheFlyDataset, pointwise_G_normalization
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class SimilarityMetric:
    def __init__(self, hparams: dict, specific_example: torch.Tensor, random_seed: int = 40, G_channels: bool = False):
        bumps = Bumps(hparams=hparams, random_seed=random_seed)
        if G_channels:
            dataset= OnTheFlyDataset(bumps=bumps,dataset_size=10000, pointwise_norm=True)
        else:
            dataset = OnTheFlyDataset(bumps=bumps,dataset_size=10000)
        self.dataloader = DataLoader(dataset, batch_size=1000, shuffle=False,num_workers=0)
        self.tau = self.dataloader.dataset.laplace.tau
        self.omega = self.dataloader.dataset.laplace.omega
        self.specific_example = specific_example.view(1, 1, -1) if len(specific_example.shape) < 3 else specific_example
        self.specific_CDF = self.specific_example.cumsum(dim=-1)
        self.specific_CDF = self.specific_CDF / self.specific_CDF[:, :, -1].unsqueeze(-1)  # Normalize the CDF

        kernel = self.dataloader.dataset.laplace.kernel
        integrand = kernel * specific_example.numpy().flatten()
        G = np.trapz(y=integrand, x=self.omega)
        A = np.trapz(y=G, x=self.tau) * np.pi
        G /= A
        enc_G = pointwise_G_normalization(mean=torch.Tensor([ 0.3319, -1.8394,  0.0023,  0.5002]), std=torch.Tensor([0.4170, 1.5327, 1.1582, 0.2884]))
        self.specific_example_G = enc_G(torch.Tensor(G).view(1,-1)).unsqueeze(0) # should have shape of (1,4,99)
        self.specific_example_G_norm = (self.specific_example_G**2).mean(-1) # should have shape of (1,4)

    def _wasserstein_distance(self, batch: torch.Tensor):
        """Compute the Wasserstein distance between a specific example and a batch of CDFs."""
        CDFs = batch.cumsum(dim=-1)
        CDFs = CDFs / CDFs[:, :, -1].unsqueeze(-1)
        return (torch.abs(CDFs - self.specific_CDF)).mean(dim=-1).squeeze(1)
    
    def _L2_distance_G(self, batch: torch.Tensor):
        """Compute the L2 distance between a specific example and a batch of G values."""
        # the batch has shape of (bs,4,99)
        # self.specific_example_G has shape of (1,4,99)
        L2 = ((batch - self.specific_example_G)**2).mean(dim=-1) # should have shape of (bs,4)
        norm_L2_per_channel =  L2 / self.specific_example_G_norm 
        return norm_L2_per_channel.mean(dim=-1) # average over channels

    def sort_similarity(self, keep_top_k: int = 20, iterations: int = 10):
        """Compute the similarity between the specific example and the dataset."""
        distance = torch.Tensor([])
        top_similar_spectra = torch.Tensor([])
        for i in tqdm(range(iterations)):
            for spectra, _ in self.dataloader:

                distance = torch.cat((distance, self._wasserstein_distance(spectra)), dim=0)
                sorted_indices = torch.argsort(distance)[:keep_top_k]

                spectra = torch.cat((top_similar_spectra, spectra), dim=0)
                top_similar_spectra = spectra[sorted_indices]
                distance = distance[sorted_indices]

        self.smallest_distances = distance
        self.top_similar_spectra = top_similar_spectra
    
    def sort_similarity_with_G(self, keep_top_k: int = 20, iterations: int = 10):
        '''
        SORTING THE SPECTRA ACCORDING TO G AND ITS CHANNELS
        '''
        distance = torch.Tensor([])
        top_similar_spectra = torch.Tensor([])
        for i in tqdm(range(iterations)):
            for spectra, G in self.dataloader:

                distance = torch.cat((distance, self._L2_distance_G(G)), dim=0)
                sorted_indices = torch.argsort(distance)[:keep_top_k]

                spectra = torch.cat((top_similar_spectra, spectra), dim=0)
                top_similar_spectra = spectra[sorted_indices]
                distance = distance[sorted_indices]

        self.smallest_distances = distance
        self.top_similar_spectra = top_similar_spectra

    def plot(self, norm_spect=True):
        n = int(np.sqrt(len(self.smallest_distances)))
        fig,axs = plt.subplots(n,n,figsize=(8.2*n/3,6*n/3),sharex=True,sharey=False)
        palette = ['C' + str(x) for x in range(250000)]
        for j, ax in enumerate(axs.flatten()):
            y = self.specific_example.numpy().flatten()
            if norm_spect:
                y /= np.trapz(y, x=self.omega)
            ax.plot(self.omega, y, color='k')
            y = self.top_similar_spectra[j].numpy().flatten()
            if norm_spect:
                y /= np.trapz(y, x=self.omega)
            ax.plot(self.omega, y, color=palette[j])
            ax.set_title(f"Distance: {self.smallest_distances[j].item():.4f}")
        plt.tight_layout()
        plt.show()

    def plot_closest_G(self):
        pointwise_mean_G = np.load('rnd_spectra/pointwise_mean_G.npy').flatten()
        pointwise_mean_abd_G = np.load('rnd_spectra/pointwise_mean_abd_G.npy').flatten()
        CDF_G = np.load('rnd_spectra/cdf_G.npy')
        possible_G_vals = np.linspace(0, 6, int(1e5))
        colors = ['k', 'tab:blue']
        specs = [self.specific_example.numpy().flatten(), self.top_similar_spectra[0].numpy().flatten()]

        fig,axs = plt.subplots(1,4,figsize=(8.2*4/3,6),sharex=True,sharey=False)
        kernel = self.dataloader.dataset.laplace.kernel
        for j in range(2):
            integrand = kernel * specs[j]
            G = np.trapz(y=integrand, x=self.omega)
            A = np.trapz(y=G, x=self.tau) * np.pi
            G /= A
            axs[0].plot(self.tau, G, color=colors[j])
            axs[1].plot(self.tau, np.log(G), color=colors[j])

            G_pointwise_norm = (G - pointwise_mean_G) / pointwise_mean_abd_G
            axs[2].plot(self.tau, G_pointwise_norm, color=colors[j])

            indices = np.searchsorted(possible_G_vals, G)
            G = np.diag(CDF_G[indices,:])
            axs[3].plot(self.tau, G, color=colors[j])
        axs[0].set_title("G")
        axs[1].set_title("log(G)")
        axs[2].set_title("Pointwise norm")
        axs[3].set_title("CDF of G")
        plt.tight_layout()
        plt.show()
