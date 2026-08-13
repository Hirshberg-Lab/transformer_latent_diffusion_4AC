#!/usr/bin/env python3
"""
Train the analytic-continuation models.

This is a standalone, faithful port of the training cells from the development
notebooks (``sharper_spectra.ipynb`` for the diffusion model and
``deterministic_NN.ipynb`` for the regression baseline).

    # Diffusion model (the generative model of the paper):
    python run_training.py --model diffusion --model-name sharper_peaks_12layers --epochs 800

    # Regression baseline (used for the Fig. 3 comparison). Repeat with several
    # --model-number values to build the ensemble reported in the paper:
    python run_training.py --model regression --model-number 1 --epochs 800

    # Multiple GPUs:
    accelerate launch run_training.py --model diffusion --model-name sharper_peaks_12layers

Notes
-----
* Training uses on-the-fly synthetic data, so an "epoch" is a fixed number of
  freshly generated batches rather than a pass over a fixed dataset. The paper
  trains for ~800 epochs on a single GPU.
* ``pointwise_norm=True`` selects the four-channel G(tau) conditioning
  (raw, log, pointwise-normalized, histogram-equalized) used throughout the
  paper. The trained checkpoints and the shipped predictions all use it.
* A CUDA GPU is strongly recommended; the model saves to ``--model-name`` so that
  ``run_inference.py`` can load it afterwards.

Run this script from the repository root.
"""

import argparse
from dataclasses import asdict

from tld.configs import (
    ModelConfig,
    DeterministicModelConfig,
    SpectralDataConfig,
    DenoiserConfig,
    DeterministicConfig,
    TrainConfig,
)


def train_diffusion(args):
    from tld.train import main as train_main

    model_name = args.model_name or "sharper_peaks_12layers"
    cfg = ModelConfig(
        # Data generator: mixtures of 1-4 warped bumps on omega in (0, 50).
        data_config=SpectralDataConfig(
            omega_domain=(0, 50),
            num_bumps_range=(1, 4),
            bump_widths_fraction_range=(0.1, 0.45),
        ),
        # 12-layer Diffusion Transformer, embedding dim 256 (see paper, Sec. VI A).
        denoiser_config=DenoiserConfig(n_layers=12, embed_dim=256),
        train_config=TrainConfig(
            compile=False,
            lr=3e-5,
            batch_size=50,
            n_epoch=args.epochs,
            use_wandb=args.wandb,
            save_model=True,
            model_name=model_name,
        ),
    )
    print(f"[run_training] diffusion model -> '{model_name}', "
          f"{args.epochs} epochs, 4-channel conditioning")
    # pointwise_norm=True => four-channel G(tau) conditioning.
    train_main(cfg, pointwise_norm=True)
    print(f"[run_training] done. Saved checkpoint: {cfg.train_config.model_name}")


def train_regression(args):
    from tld.train_deterministic_model import main as train_main

    model_name = args.model_name or f"deterministic_model_{args.model_number}"
    cfg = DeterministicModelConfig(
        data_config=SpectralDataConfig(
            omega_domain=(0, 50),
            num_bumps_range=(1, 4),
            bump_widths_fraction_range=(0.1, 0.6),
            bump_centers_fraction_range=(0.0, 0.5),
        ),
        deterministic_config=DeterministicConfig(n_layers=12, embed_dim=256),
        train_config=TrainConfig(
            compile=False,
            lr=3e-5,
            batch_size=50,
            n_epoch=args.epochs,
            use_wandb=args.wandb,
            save_model=True,
            model_name=model_name,
        ),
    )
    print(f"[run_training] regression baseline -> '{model_name}', {args.epochs} epochs")
    train_main(cfg, pointwise_norm=True)
    print(f"[run_training] done. Saved checkpoint: {model_name}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["diffusion", "regression"], default="diffusion",
                   help="Which model to train (default: diffusion).")
    p.add_argument("--epochs", type=int, default=800,
                   help="Number of on-the-fly epochs (default: 800, as in the paper).")
    p.add_argument("--model-name", default=None,
                   help="Filename for the saved checkpoint (diffusion default: "
                        "sharper_peaks_12layers; regression default: "
                        "deterministic_model_<model-number>).")
    p.add_argument("--model-number", default="1",
                   help="Ensemble index for the regression baseline (default: 1).")
    p.add_argument("--wandb", action="store_true",
                   help="Log the run to Weights & Biases (requires a wandb login).")
    return p


def main():
    args = build_parser().parse_args()
    if args.model == "diffusion":
        train_diffusion(args)
    else:
        train_regression(args)


if __name__ == "__main__":
    main()
