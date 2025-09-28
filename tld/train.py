#!/usr/bin/env python3

import copy
from dataclasses import asdict

import numpy as np
import torch
# import torchvision
# import torchvision.utils as vutils
try:
    import wandb
except:
    print('please install Weights & Biases')
from accelerate import Accelerator
# from diffusers import AutoencoderKL
# from PIL.Image import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from tld.denoiser import Denoiser
from tld.diffusion import DiffusionGenerator
from tld.configs import ModelConfig

from rnd_spectra.bumps import Bumps
from rnd_spectra.ontheflydataset import OnTheFlyDataset


# def eval_gen(diffuser: DiffusionGenerator, labels: Tensor, img_size: int) -> Image:
#     class_guidance = 4.5
#     seed = 10
#     out, _ = diffuser.generate(
#         labels=torch.repeat_interleave(labels, 2, dim=0),
#         num_imgs=16,
#         class_guidance=class_guidance,
#         seed=seed,
#         n_iter=40,
#         exponent=1,
#         sharp_f=0.1,
#         img_size=img_size
#     )

#     out = to_pil((vutils.make_grid((out + 1) / 2, nrow=8, padding=4)).float().clip(0, 1))
#     out.save(f"emb_val_cfg:{class_guidance}_seed:{seed}.png")

#     return out


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_per_layer(model: nn.Module):
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")


# to_pil = torchvision.transforms.ToPILImage()


def update_ema(ema_model: nn.Module, model: nn.Module, alpha: float = 0.999):
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(model_param.data, alpha=1 - alpha)



def main(config: ModelConfig, use_stft: bool = False, pointwise_norm: bool = False) -> Denoiser:
    """main train loop to be used with accelerate"""
    denoiser_config = config.denoiser_config
    train_config = config.train_config
    dataconfig = config.data_config

    log_with="wandb" if train_config.use_wandb else None
    accelerator = Accelerator(mixed_precision="fp16", log_with=log_with)

    accelerator.print("Creating training loader:")
    hparams = asdict(dataconfig)
    random_seed = 40
    bumps = Bumps(hparams=hparams, random_seed=random_seed)
    dataset = OnTheFlyDataset(bumps=bumps, dataset_size=train_config.dataset_size, use_stft=use_stft, pointwise_norm=pointwise_norm)
    train_loader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=False,num_workers=0)

    model = Denoiser(**asdict(denoiser_config))

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)

    if train_config.compile:
        accelerator.print("Compiling model:")
        model = torch.compile(model)

    if not train_config.from_scratch:
        accelerator.print("Loading Model:")
        wandb.restore(
            train_config.model_name, run_path=f"sagimeir-tel-aviv-university/Diffusion_AC/runs/{train_config.run_id}", replace=True
        )
        full_state_dict = torch.load(train_config.model_name)
        model.load_state_dict(full_state_dict["model_ema"])
        optimizer.load_state_dict(full_state_dict["opt_state"])
        global_step = full_state_dict["global_step"]
    else:
        global_step = 0

    if accelerator.is_local_main_process:
        ema_model = copy.deepcopy(model).to(accelerator.device)

    accelerator.print("model prep")
    model, train_loader, optimizer = accelerator.prepare(model, train_loader, optimizer)

    if train_config.use_wandb:
        accelerator.init_trackers(project_name="Diffusion_AC", config=asdict(config))

    accelerator.print(f"The model has {count_parameters(model)} parameters")
    # accelerator.print(f"Now printing parameters per layer:\n{count_parameters_per_layer(model)}")

    T = 1000
    alphas = torch.linspace(start=0.9999, end=0.98, steps=T, dtype=torch.float64, device=accelerator.device)
    alpha_bars = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bars_t = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars_t = torch.sqrt(1.0 - alpha_bars)

    ### Train:
    epoch_loss=[]
    for i in range(1, train_config.n_epoch + 1):
        # accelerator.print(f"epoch: {i}")
        batch_loss=[]
        for x_0, y in tqdm(train_loader):

            time = torch.randint(0, T, (x_0.size(0),), device=accelerator.device)

            noise = torch.randn_like(x_0)

            x_t = sqrt_alpha_bars_t[time].view(-1, 1, 1) * x_0 + sqrt_one_minus_alpha_bars_t[time].view(-1, 1, 1) * noise

            x_t = x_t.float()
            time = time.float()
            label = y

            prob = 0.15
            mask = torch.rand(y.size(0), device=accelerator.device) < prob
            label[mask] = 0  # OR replacement_vector

            if global_step % train_config.save_and_eval_every_iters == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ##eval and saving:
                    # out = eval_gen(diffuser=diffuser, labels=emb_val, img_size=denoiser_config.x_points)
                    # out.save("img.jpg")
                    # if train_config.use_wandb:
                    #     accelerator.log({f"step: {global_step}": wandb.Image("img.jpg")}) # Note: I will change this line in the future

                    # opt_unwrapped = accelerator.unwrap_model(optimizer) 
                    opt_unwrapped = optimizer # Note: the line above did not work - apparently the optimizer is not "wrapped" so there is no need to unwrap it
                    full_state_dict = {
                        "model_ema": ema_model.state_dict(),
                        "opt_state": opt_unwrapped.state_dict(),
                        "global_step": global_step,
                    }
                    if train_config.save_model:
                        accelerator.save(full_state_dict, train_config.model_name)
                        if train_config.use_wandb:
                            wandb.save(train_config.model_name)

            model.train()

            with accelerator.accumulate():
                ###train loop:
                optimizer.zero_grad()

                pred = model(x_t, time.view(-1, 1), label)
                loss = loss_fn(pred, noise)
                # accelerator.log({"train_loss": loss.item()}, step=global_step)
                accelerator.backward(loss)
                optimizer.step()
                batch_loss.append(loss.item())

                if accelerator.is_main_process:
                    update_ema(ema_model, model, alpha=train_config.alpha)

            global_step += 1
        ep_loss = np.mean(batch_loss)
        epoch_loss.append(ep_loss)
        accelerator.print(f"epoch: {i} | train_loss: {ep_loss}")
        accelerator.log({"train_loss": ep_loss}, step=global_step)
    accelerator.end_training()
    return ema_model


# args = (config, data_path, val_path)
# notebook_launcher(training_loop)
