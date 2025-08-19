from rnd_spectra.bumps import Bumps
from rnd_spectra.laplace import Laplace
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Optional
from einops.layers.torch import Rearrange

class OnTheFlyDataset(Dataset):
    '''
    Uses the classes Bumps and Laplace to generate a specturm (C) and its corresponding iTCF (G).
    C is the ground truth result of iverting G.
    '''
    def __init__(self, bumps: Bumps, 
                 dataset_size: int = 10000, 
                 dtype: str = 'float32',
                 normalize_inputs: Optional[tuple[float,float]] = (0.,1.),
                 normalize_labels: Optional[tuple[float,float]] = (0.,1.),
                 use_stft: bool = False,
                 pointwise_norm: bool = False,
                 channel_mean: torch.Tensor = torch.Tensor([ 0.3319, -1.8394,  0.0023,  0.5002]),
                 channel_std: torch.Tensor = torch.Tensor([0.4170, 1.5327, 1.1582, 0.2884])
                 ) -> None:
        
        self.laplace = Laplace(bumps)
        self.dataset_size = dataset_size
        self.dtype = dtype

        # for normalization
        self.mean_G ,self.std_G = normalize_labels
        self.mean_C ,self.std_C = normalize_inputs

        if use_stft:
            self.encoding_G = STFT(hop_length=1)
            self.encoding_C = STFT()
        elif pointwise_norm:
            self.encoding_G = pointwise_G_normalization(mean=channel_mean, std=channel_std)
            self.encoding_C = nn.Identity()
        else:
            self.encoding_G = nn.Identity()
            self.encoding_C = nn.Identity()

    def __len__(self) -> None:
        return self.dataset_size

    def __getitem__(self, idx) -> tuple[torch.Tensor,torch.Tensor]:
        G = self.laplace.evaluate_transformation() # the labels
        C = self.laplace.spectrum 
        
        G = (G-self.mean_G)/self.std_G
        C = (C-self.mean_C)/self.std_C
        
        G = torch.from_numpy(G.astype(self.dtype)).unsqueeze(0)
        C = torch.from_numpy(C.astype(self.dtype)).unsqueeze(0)

        return self.encoding_C(C), self.encoding_G(G)

class pointwise_G_normalization(nn.Module):
    def __init__(self, pointwise_mean_G: np.array = np.load('rnd_spectra/pointwise_mean_G.npy'), 
                 pointwise_mean_abd_G: np.array = np.load('rnd_spectra/pointwise_mean_abd_G.npy'), 
                 CDF: np.array = np.load('rnd_spectra/cdf_G.npy'), 
                 possible_G_vals: torch.Tensor = torch.linspace(0,6,int(1e5)),
                 mean: torch.Tensor = torch.Tensor([ 0, 0, 0,  0]),
                 std: torch.Tensor = torch.Tensor([1, 1, 1, 1])
                 ) -> None:
        super().__init__()
        self.pointwise_mean_G = torch.from_numpy(pointwise_mean_G)
        self.pointwise_mean_abd_G = torch.from_numpy(pointwise_mean_abd_G)
        self.CDF = torch.from_numpy(CDF)
        self.possible_G_vals = possible_G_vals
        self.mean = mean[None,:, None]
        self.std = std[None,:, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pointwise_norm = (x - self.pointwise_mean_G)/ self.pointwise_mean_abd_G
        indices = torch.searchsorted(self.possible_G_vals, x.squeeze(0))
        x_cdf = torch.diag(self.CDF[indices,:]).unsqueeze(0)
        G = torch.cat([x, x.log(), x_pointwise_norm, x_cdf], dim=0)
        G = (G - self.mean) / self.std
        return G.squeeze(0)

class STFT(nn.Module):
    def __init__(self, n_fft: int = 90,
                 hop_length: int = 4, 
                 win_length: int = 5
                 ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)
        self.rearrange = Rearrange(
            "b c l z -> b c (z l)",
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, x_points)
        x= torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )
        x = torch.view_as_real(x)
        return self.rearrange(x).squeeze(0)

class ISTFT(nn.Module):
    def __init__(self, n_fft: int = 90, 
                 hop_length: int = 4, 
                 win_length: int = 5,
                 length: int = 1024
                 ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)
        self.length = length
        self.rearrange = Rearrange(
            "b c (z l) -> b c l z",
            z=2,
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, x_points)
        x = self.rearrange(x).contiguous() # this way all dimensions of returned tensor will have stride 1. see https://stackoverflow.com/questions/63852258/how-can-i-get-a-view-of-input-as-a-complex-tensor-runtimeerror-tensor-must-hav
        x = torch.view_as_complex(x)
        return torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=False,
            length=self.length
        ) 
