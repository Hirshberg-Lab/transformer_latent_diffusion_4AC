from rnd_spectra.bumps import Bumps
from rnd_spectra.laplace import Laplace
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional

class OnTheFlyDataset(Dataset):
    '''
    Uses the classes Bumps and Laplace to generate a specturm (C) and its corresponding iTCF (G)
    C is the ground truth label of G.
    '''
    def __init__(self, bumps: Bumps, 
                 dataset_size: int = 10000, 
                 dtype: str = 'float32',
                 normalize_inputs: Optional[tuple[float,float]] = (0.,1.),
                 normalize_labels: Optional[tuple[float,float]] = (0.,1.)
                 ) -> None:
        
        self.laplace = Laplace(bumps)
        self.dataset_size = dataset_size
        self.dtype = dtype

        # for normalization
        self.mean_G ,self.std_G = normalize_labels
        self.mean_C ,self.std_C = normalize_inputs

    def __len__(self) -> None:
        return self.dataset_size

    def __getitem__(self, idx) -> tuple[torch.Tensor,torch.Tensor]:
        G = self.laplace.evaluate_transformation() # the labels
        C = self.laplace.spectrum 
        G = (G-self.mean_G)/self.std_G
        C = (C-self.mean_C)/self.std_C
        return torch.from_numpy(C.astype(self.dtype)), torch.from_numpy(G.astype(self.dtype))
