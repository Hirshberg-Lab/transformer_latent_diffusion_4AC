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
                 stft_C_mean: torch.Tensor = torch.Tensor([1.9e-02, -1.4e-06]),
                 stft_C_std: torch.Tensor = torch.Tensor([1.8, 0.16]),
                 stft_G_mean: torch.Tensor = torch.Tensor([2.59e-02, -1.07e-05]),
                 stft_G_std: torch.Tensor = torch.Tensor([1.4, 0.338])
                 ) -> None:
        
        self.laplace = Laplace(bumps)
        self.dataset_size = dataset_size
        self.dtype = dtype

        # for normalization
        self.mean_G ,self.std_G = normalize_labels
        self.mean_C ,self.std_C = normalize_inputs

        if use_stft:
            self.encoding_G = STFT(hop_length=1,mean=stft_G_mean,std=stft_G_std)
            self.encoding_C = STFT(mean=stft_C_mean,std=stft_C_std)
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



class STFT(nn.Module):
    def __init__(self, n_fft: int = 90,
                 hop_length: int = 9, 
                 win_length: int = 14,
                 mean: torch.Tensor = torch.Tensor([0, 0]),
                 std: torch.Tensor = torch.Tensor([1, 1])
                 ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)
        self.rearrange = Rearrange(
            "b freq time z -> b z freq time",
        )
        self.mean = mean[None, :, None, None]
        self.std = std[None, :, None, None]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x= torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )
        x = torch.view_as_real(x)
        x = self.rearrange(x)
        x = (x - self.mean) / self.std
        return x.squeeze(0)

class ISTFT(nn.Module):
    def __init__(self, n_fft: int = 90, 
                 hop_length: int = 9, 
                 win_length: int = 14,
                 length: int = 1024,
                 mean: torch.Tensor = torch.Tensor([0, 0]),
                 std: torch.Tensor = torch.Tensor([1, 1])
                 ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)
        self.length = length
        self.rearrange = Rearrange(
            "b z freq time -> b freq time z",
            z=2,
        )
        self.mean = mean[None, :, None, None]
        self.std = std[None, :, None, None]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x * self.std) + self.mean        
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
