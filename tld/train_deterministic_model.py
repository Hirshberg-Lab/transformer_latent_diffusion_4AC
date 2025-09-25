import copy
from dataclasses import asdict

import numpy as np
import torch
try:
    import wandb
except:
    print('please install Weights & Biases')
from accelerate import Accelerator

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tld.deterministic import Deterministic_NN
from tld.configs import DeterministicModelConfig

from rnd_spectra.bumps import Bumps
from rnd_spectra.ontheflydataset import OnTheFlyDataset

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def update_ema(ema_model: nn.Module, model: nn.Module, alpha: float = 0.999):
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(model_param.data, alpha=1 - alpha)

def main(config: DeterministicModelConfig, use_stft: bool = False, pointwise_norm: bool = False) -> Deterministic_NN:
    """main train loop to be used with accelerate"""
    deterministic_config = config.deterministic_config
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

    model = Deterministic_NN(**asdict(deterministic_config))

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)

    if not train_config.from_scratch:
        accelerator.print("Loading Model:")
        wandb.restore(
            train_config.model_name, run_path=f"sagimeir-tel-aviv-university/Reg_AC/runs/{train_config.run_id}", replace=True
        )
        full_state_dict = torch.load(train_config.model_name)
        model.load_state_dict(full_state_dict["model_ema"])
        optimizer.load_state_dict(full_state_dict["opt_state"])
        global_step = full_state_dict["global_step"]
    else:
        global_step = 0

    if accelerator.is_local_main_process:
        ema_model = copy.deepcopy(model).to(accelerator.device)
        # diffuser = DiffusionGenerator(ema_model, accelerator.device, torch.float32)

    accelerator.print("model prep")
    model, train_loader, optimizer = accelerator.prepare(model, train_loader, optimizer)

    if train_config.use_wandb:
        accelerator.init_trackers(project_name="Reg_AC", config=asdict(config))

    accelerator.print(f"The model has {count_parameters(model)} parameters")

    model.train()
    epoch_loss=[]
    for i in range(1, train_config.n_epoch + 1):
        batch_loss=[]
        for x, y in tqdm(train_loader):
                        
            if global_step % train_config.save_and_eval_every_iters == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    opt_unwrapped = optimizer
                    full_state_dict = {
                        "model_ema": ema_model.state_dict(),
                        "opt_state": opt_unwrapped.state_dict(),
                        "global_step": global_step,
                    }
                    if train_config.save_model:
                        accelerator.save(full_state_dict, train_config.model_name)
                        if train_config.use_wandb:
                            wandb.save(train_config.model_name)
            
            with accelerator.accumulate():
                pred = model(y)
                loss = loss_fn(pred, x)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
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