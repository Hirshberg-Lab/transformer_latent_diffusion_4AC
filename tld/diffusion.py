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
        n_iter: int = 30,
        num_specs: int = 16,
        class_guidance: float = 3,
        seed: int = 10,
        scale_factor: int = 8,  # latent scaling before decoding - should be ~ std of latent space
        x_points: int = 1024,  # 
        sharp_f: float = 0.1,
        bright_f: float = 0.1,
        exponent: float = 1,
        seeds: Tensor | None = None, # It looks that this should always be None
        noise_levels=None,
        use_ddpm_plus: bool = True,
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

        if noise_levels is None:
            noise_levels = (1 - torch.pow(torch.arange(0, 1, 1 / n_iter), exponent)).tolist()
        noise_levels[0] = 0.99

        if use_ddpm_plus:
            lambdas = [np.log((1 - sigma) / sigma) for sigma in noise_levels]  # log snr
            hs = [lambdas[i] - lambdas[i - 1] for i in range(1, len(lambdas))]
            rs = [hs[i - 1] / hs[i] for i in range(1, len(hs))]

        x_t = self.initialize_spectrum(seeds, num_specs, x_points, seed)

        labels = torch.cat([labels, torch.zeros_like(labels)])
        self.model.eval()

        x0_pred_prev = None
        class_guidance = [1 - (1-class_guidance) * n/len(noise_levels) for n in range(len(noise_levels))]
        constraint_guidance = [1 - (1-constraint_guidance) * n/len(noise_levels) for n in range(len(noise_levels))]
        # easy to see that if constraint_guidance=1, then this is just 1 for all n, i.e., [1, 1, ..., 1]

        for i in tqdm(range(len(noise_levels) - 1)):
            curr_noise, next_noise = noise_levels[i], noise_levels[i + 1]

            x0_pred = self.pred_spec(x_t, labels, curr_noise, class_guidance[i], 
                                     constraint_guidance[i], prior_knowledge, directed_positivity, kernel)

            if x0_pred_prev is None:
                x_t = ((curr_noise - next_noise) * x0_pred + next_noise * x_t) / curr_noise
            else:
                if use_ddpm_plus:
                    # x0_pred is a combination of the two previous x0_pred:
                    D = (1 + 1 / (2 * rs[i - 1])) * x0_pred - (1 / (2 * rs[i - 1])) * x0_pred_prev
                else:
                    # ddim:
                    D = x0_pred

                x_t = ((curr_noise - next_noise) * D + next_noise * x_t) / curr_noise

            x0_pred_prev = x0_pred

        x0_pred = self.pred_spec(x_t, labels, next_noise, class_guidance[i+1], 
                                 constraint_guidance[i+1], prior_knowledge, directed_positivity, kernel)

        # shifting latents works a bit like an image editor:
        # x0_pred[:, 3, :, :] += sharp_f
        # x0_pred[:, 0, :, :] += bright_f

        # x0_pred_img = self.vae.decode((x0_pred * scale_factor).to(self.model_dtype))[0].cpu()
        return x0_pred

    def pred_spec(self, noisy_spec, labels, noise_level, class_guidance, 
                  constraint_guidance, prior_knowledge, directed_positivity, kernel):
        num_specs = noisy_spec.size(0)
        noises = torch.full((2 * num_specs, 1), noise_level)
        x0_pred = self.model(
            torch.cat([noisy_spec, noisy_spec]),
            noises.to(self.device, self.model_dtype),
            labels.to(self.device, self.model_dtype),
        )
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
                # img_size,
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


def download_file(url, filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


# @torch.no_grad()
# def encode_text(label, model):
#     text_tokens = clip.tokenize(label, truncate=True).to(device)
#     text_encoding = model.encode_text(text_tokens)
#     return text_encoding.cpu()


# class DiffusionTransformer:
#     def __init__(self, cfg: LTDConfig):
#         denoiser = Denoiser(**asdict(cfg.denoiser_cfg))
#         denoiser = denoiser.to(cfg.denoiser_load.dtype)

#         if cfg.denoiser_load.file_url is not None:
#             if cfg.denoiser_load.local_filename is not None:
#                 print(f"Downloading model from {cfg.denoiser_load.file_url}")
#                 download_file(cfg.denoiser_load.file_url, cfg.denoiser_load.local_filename)
#                 state_dict = torch.load(cfg.denoiser_load.local_filename, map_location=torch.device("cpu"))
#                 denoiser.load_state_dict(state_dict)

#         denoiser = denoiser.to(device)

#         # vae = AutoencoderKL.from_pretrained(cfg.vae_cfg.vae_name, 
#         # torch_dtype=cfg.vae_cfg.vae_dtype).to(device)

#         # self.clip_model, preprocess = clip.load(cfg.clip_cfg.clip_model_name)
#         # self.clip_model = self.clip_model.to(device)

#         self.diffuser = DiffusionGenerator(denoiser, device, cfg.denoiser_load.dtype)

#     def generate_image_from_text(
#         self, prompt: str, class_guidance=6, seed=11, num_imgs=1, img_size=32, n_iter=15
#     ):
#         # nrow = int(np.sqrt(num_imgs))

#         cur_prompts = [prompt] * num_imgs
#         # labels = encode_text(cur_prompts, self.clip_model)
#         out = self.diffuser.generate(
#             labels=labels,
#             num_specs=num_imgs,
#             x_points=self.diffuser.model.x_points,
#             class_guidance=class_guidance,
#             seed=seed,
#             n_iter=n_iter,
#             exponent=1,
#             scale_factor=8,
#             sharp_f=0,
#             bright_f=0,
#         )

#         # out = to_pil((vutils.make_grid((out + 1) / 2, nrow=nrow, padding=4)).float().clip(0, 1))
#         return out
