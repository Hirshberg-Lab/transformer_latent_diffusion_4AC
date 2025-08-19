from rnd_spectra.bumps import Bumps
from rnd_spectra.ontheflydataset import OnTheFlyDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class SimilarityMetric:
    def __init__(self, hparams: dict, specific_example: torch.Tensor, random_seed: int = 40):
        bumps = Bumps(hparams=hparams, random_seed=random_seed)
        dataset = OnTheFlyDataset(bumps=bumps,dataset_size=10000)
        self.dataloader = DataLoader(dataset, batch_size=1000, shuffle=False,num_workers=0)
        self.tau = self.dataloader.dataset.laplace.tau
        self.omega = self.dataloader.dataset.laplace.omega
        self.specific_example = specific_example.view(1, 1, -1) if len(specific_example.shape) < 3 else specific_example
        self.specific_CDF = self.specific_example.cumsum(dim=-1)
        self.specific_CDF = self.specific_CDF / self.specific_CDF[:, :, -1].unsqueeze(-1)  # Normalize the CDF

    def _wasserstein_distance(self, batch: torch.Tensor):
        """Compute the Wasserstein distance between a specific example and a batch of CDFs."""
        CDFs = batch.cumsum(dim=-1)
        CDFs = CDFs / CDFs[:, :, -1].unsqueeze(-1)
        return (torch.abs(CDFs - self.specific_CDF)).mean(dim=-1).squeeze(1)

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

    def plot(self):
        n = int(np.sqrt(len(self.smallest_distances)))
        fig,axs = plt.subplots(n,n,figsize=(8.2*n/3,6*n/3),sharex=True,sharey=False)
        palette = ['C' + str(x) for x in range(250000)]
        for j, ax in enumerate(axs.flatten()):
            ax.plot(self.omega, self.specific_example.numpy().flatten(), color='k')
            ax.plot(self.omega, self.top_similar_spectra[j].numpy().flatten(), color=palette[j])
            ax.set_title(f"Distance: {self.smallest_distances[j].item():.4f}")
        plt.tight_layout()
        plt.show()
