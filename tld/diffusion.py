from dataclasses import dataclass, asdict
from typing import Optional
# import clip
import numpy as np
import requests
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
# from diffusers import AutoencoderKL
from torch import Tensor
from tqdm import tqdm

from tld.denoiser import Denoiser

# from tld.configs import LTDConfig


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
to_pil = transforms.ToPILImage()


@dataclass
class DiffusionGenerator:
    model: Denoiser
    # vae: AutoencoderKL
    device: torch.device
    model_dtype: torch.dtype = torch.float32

    @torch.no_grad()
    def generate(
        self,
        labels: Tensor,  # embeddings to condition on
        n_iter: int = 1000,
        num_specs: int = 16,
        class_guidance: float = 3,
        seed: int = 10,
        x_points: int = 1024,  # 
        seeds: Tensor | None = None, # It looks that this should always be None
        use_ddpm_plus: bool = False,
        constraint_guidance: float = 1, # between 0 and 1, 1 means no constraint, 0 means only the constraint
        prior_knowledge: float | Tensor = 0,
        directed_positivity: bool = False,
        omega: Optional[np.array] = None,
        tau: Optional[np.array] = None,
        # if omega and tau are NOT None, then they are used to compute the prior knowledge
    ):
        """Generate images via reverse diffusion.
        if use_ddpm_plus=True uses Algorithm 2 DPM-Solver++(2M) here: https://arxiv.org/pdf/2211.01095.pdf
        else use ddim with alpha = 1-sigma
        """
        kernel = None
        if omega is not None and tau is not None:
            kernel = ( \
                torch.exp( -torch.Tensor(omega * tau[:,None]) ) + \
                      torch.exp( torch.Tensor(omega * ( tau[:,None] -1 )) ) \
                    )/2./np.pi*omega[1]
            kernel = kernel.to(self.device, self.model_dtype)
        
        T = n_iter
        alphas = torch.linspace(start=0.9999, end=0.98, steps=T, dtype=torch.float64, device=self.device)
        alpha_bars = torch.cumprod(alphas, dim=0)
        sqrt_one_minus_alpha_bars_t = torch.sqrt(1.0 - alpha_bars)
        alpha_bars_prev = torch.cat((torch.ones(1).to(self.device), alpha_bars[:-1]))
        sigma_t_squared = (1.0 - alphas) * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        sigma_t = torch.sqrt(sigma_t_squared)

        
        x_t = self.initialize_spectrum(seeds, num_specs, x_points, seed)

        labels = torch.cat([labels, torch.zeros_like(labels)])
        self.model.eval()

        x0_pred_prev = None
        class_guidance = [1 - (1-class_guidance) * n/n_iter for n in range(n_iter)]
        constraint_guidance = [1 - (1-constraint_guidance) * n/n_iter for n in range(n_iter)]
        # easy to see that if constraint_guidance=1, then this is just 1 for all n, i.e., [1, 1, ..., 1]

        for t in tqdm(reversed(range(1,T)),total=T-1):
            x_t = self.pred_spec(x_t, labels, 
                                 t, alphas[t], sqrt_one_minus_alpha_bars_t[t], sigma_t[t],
                                 class_guidance[t], constraint_guidance[t], prior_knowledge, directed_positivity, kernel
                                 ).float()
        
        x0_pred = self.pred_spec(x_t, labels, 
                                 t, alphas[0], sqrt_one_minus_alpha_bars_t[0], 0.,
                                 class_guidance[t], constraint_guidance[t], prior_knowledge, directed_positivity, kernel
                                 ).float()

        return x0_pred

    def pred_spec(self, x_t, labels, 
                  t, alpha, sqrt_one_minus_alpha_bar, sigma,
                  class_guidance, constraint_guidance, prior_knowledge, directed_positivity, kernel):
        num_specs = x_t.size(0)
        time = torch.full((2 * num_specs, 1), t).float()
        x_t = torch.cat([x_t, x_t])
        eps = self.model(
            x_t,
            time.to(self.device, self.model_dtype),
            labels.to(self.device, self.model_dtype),
        )
        x0_pred = (1.0 / torch.sqrt(alpha)) * (x_t - ((1.0 - alpha) / sqrt_one_minus_alpha_bar) * eps) + \
                sigma * torch.randn_like(eps)
        if kernel is not None:
            prior_knowledge = x0_pred[:num_specs] + \
                  10 * (labels[:num_specs] - x0_pred[:num_specs] @ kernel.T) @ kernel
            prior_knowledge = torch.relu(prior_knowledge)
        elif directed_positivity:
            prior_knowledge = torch.relu(x0_pred[:num_specs])
        x0_pred = self.apply_classifier_free_guidance(x0_pred, num_specs, class_guidance, constraint_guidance, prior_knowledge)
        return x0_pred

    def initialize_spectrum(self, seeds, num_specs, x_points, seed):
        """Initialize the seed tensor."""
        if seeds is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            return torch.randn(
                num_specs,
                self.model.n_channels,
                x_points,
                dtype=self.model_dtype,
                device=self.device,
                generator=generator,
            )
        else:
            return seeds.to(self.device, self.model_dtype)

    def apply_classifier_free_guidance(self, x0_pred, num_specs, class_guidance, constraint_guidance=1, prior_knowledge=0):
        """Apply classifier-free guidance to the predictions."""
        x0_pred_label, x0_pred_no_label = x0_pred[:num_specs], x0_pred[num_specs:]
        # return class_guidance * x0_pred_label + (1 - class_guidance) * x0_pred_no_label
        return constraint_guidance * class_guidance * x0_pred_label \
            + constraint_guidance * (1 - class_guidance) * x0_pred_no_label \
            + (1-constraint_guidance) * class_guidance * prior_knowledge

