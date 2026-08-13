# Diffusion Models for Uncertainty Quantification in Analytic Continuation

A conditional **diffusion-model** framework for the numerical **analytic continuation** of
imaginary-time correlation functions (iTCFs) `G(τ)` to real-frequency power spectra `C(ω)`.
Instead of predicting a single spectrum, the model learns the full conditional distribution
`p(C | G)` of spectra consistent with a given iTCF, which lets us (i) quantify the
uncertainty of the inversion directly from the learned distribution and (ii) measure the
intrinsic *hardness* of each inversion with a new metric, the **Uncertainty Pseudo-Volume (UPV)**.

This repository contains the code, the synthetic-data generator, and precomputed model
predictions needed to reproduce the figures of the paper:

> **Using Diffusion Models to Estimate Uncertainties in Analytic Continuation**
> Sagi Meir, Daniel Freedman, and Barak Hirshberg.

> ℹ️ Update the citation block (`CITATION.cff`) and the link above with the arXiv / journal
> reference once it is available.

The model architecture is a 1D **Diffusion Transformer (DiT)** adapted from
[`apapiu/transformer_latent_diffusion`](https://github.com/apapiu/transformer_latent_diffusion)
(MIT, © 2023 Alexandru Papiu); see [NOTICE](NOTICE) for attribution details.

---

## Method in one paragraph

Random physical spectra `C(ω)` are generated on the fly as mixtures of smoothly warped
"bumps" and mapped to their iTCFs `G(τ)` through the forward (smoothing) transform. A
Diffusion Transformer is trained with a flow-matching / continuous-time interpolant objective
to denoise `C(ω)` conditioned on a **four-channel representation** of `G(τ)` (raw, log,
pointwise-normalized, and histogram-equalized). At inference we integrate the reverse ODE with
a DPM-Solver++ sampler and draw an ensemble of spectra per iTCF; the spread of that ensemble is
the inversion uncertainty, and a local PCA of the ensemble yields the UPV.

## Repository layout

```
.
├── run_training.py              # train the diffusion model (or the regression baseline)
├── run_inference.py             # regenerate the prediction .npz files consumed by the figures
├── tld/                         # model + training library
│   ├── denoiser.py              #   Diffusion Transformer (DiT) denoiser
│   ├── transformer_blocks.py    #   attention / MLP / embedding building blocks
│   ├── diffusion.py             #   DiffusionGenerator: reverse-ODE sampler (DPM-Solver++)
│   ├── deterministic.py         #   regression baseline (same DiT core, single output)
│   ├── train.py                 #   diffusion training loop (accelerate)
│   ├── train_deterministic_model.py
│   └── configs.py               #   dataclass configs
├── rnd_spectra/                 # on-the-fly synthetic data generator
│   ├── bumps.py                 #   random spectra C(ω) (mixtures of warped Gaussians)
│   ├── laplace.py               #   forward transform  C(ω) -> G(τ)
│   ├── ontheflydataset.py       #   Dataset + 4-channel G(τ) conditioning
│   └── *.npy                    #   precomputed conditioning statistics (required)
├── data/                        # physical inputs for the parahydrogen example
│   ├── Rabani_D_w_14K.csv        #   maximum-entropy reference spectrum
│   └── barak/                    #   PIMD / Rabani iTCFs G(τ)
├── spectra/                     # precomputed model predictions (so figures need no GPU)
├── create_figs/                 # one script per paper figure (see mapping below)
└── figs/                        # figure outputs are written here
```

## Installation

Requires Python ≥ 3.10 and (for training / inference) a CUDA GPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> ⚠️ **NumPy < 2.0 is required.** The code uses `numpy.trapz`, which was removed in NumPy 2.0.
> This pin is already in `requirements.txt`.

All scripts are meant to be run **from the repository root** (paths such as `data/…`,
`spectra/…`, `rnd_spectra/…` are resolved relative to it).

## Quickstart — reproduce the figures

The trained model weights (~136 MB each) are **not** included in the repository. However, the
precomputed model **predictions** are shipped under `spectra/`, so every figure can be
reproduced with no GPU:

```bash
python create_figs/ill_posed_demo.py          # -> figs/ill_posed_demo.pdf
python create_figs/four_channel_example.py     # -> figs/four_channel_example.pdf
python create_figs/comparison3.py              # -> figs/two_rows.pdf
python create_figs/visualize_convergence.py    # -> figs/visualize_convergence.pdf
python create_figs/pseudo_volume_bar_chart2.py # -> figs/pseudo_volume_bar_chart2.pdf
python create_figs/parahydrogen.py             # -> figs/parahydrogen.pdf
python create_figs/visualize_components.py     # -> figs/visualize_components.pdf
```

### Figure → script map

| Paper figure | Script | Output |
|---|---|---|
| Fig 1 — ill-posedness demo | `create_figs/ill_posed_demo.py` | `figs/ill_posed_demo.pdf` |
| Fig 2 — four-channel conditioning | `create_figs/four_channel_example.py` | `figs/four_channel_example.pdf` |
| Fig 3 — regression vs. diffusion | `create_figs/comparison3.py` | `figs/two_rows.pdf` |
| Fig 4 — PCA reconstruction / parsimony | `create_figs/visualize_convergence.py` | `figs/visualize_convergence.pdf` |
| Fig 5 — Uncertainty Pseudo-Volume | `create_figs/pseudo_volume_bar_chart2.py` | `figs/pseudo_volume_bar_chart2.pdf` |
| Fig 6 — parahydrogen application | `create_figs/parahydrogen.py` | `figs/parahydrogen.pdf` |
| Fig S1 — principal components | `create_figs/visualize_components.py` | `figs/visualize_components.pdf` |

`create_figs/uncertainty_analysis.py` (PCA/parsimony analysis) and
`create_figs/pseudo_volume_bar_chart.py` (UPV helper) are imported by the scripts above and are
not run directly.

## Full reproduction — train from scratch

The full pipeline is **train → run inference → make figures**. Steps 1–2 require a GPU.

**1. Train the diffusion model** (~800 epochs of on-the-fly synthetic data in the paper):

```bash
python run_training.py --model diffusion --model-name sharper_peaks_12layers --epochs 800
# multi-GPU: accelerate launch run_training.py --model diffusion ...
```

Train the regression baseline used in Fig 3 (repeat with several model numbers to form the
ensemble shown in the paper):

```bash
python run_training.py --model regression --model-number 1 --epochs 800
```

**2. Regenerate the predictions** consumed by the figures (writes into `spectra/`):

```bash
python run_inference.py --model diffusion  --model-name sharper_peaks_12layers   # -> data_with_diffusion.npz, diffusion_rabani_pred.npz
python run_inference.py --model regression --model-number 1                       # -> data_deterministic_1.npz
```

**3. Make the figures** as in the Quickstart above.

Data generation, model, and sampling hyperparameters follow the paper (see `tld/configs.py`
and the constants at the top of `run_training.py` / `run_inference.py`).

## Data

- `rnd_spectra/` generates synthetic `(C(ω), G(τ))` pairs on the fly; the `.npy` files are
  precomputed conditioning statistics (pointwise mean/MAD and the empirical CDF) required by the
  four-channel representation.
- `data/barak/G_*_14.0K_180p.npy` are the imaginary-time correlation functions from the
  path-integral molecular-dynamics (PIMD) simulation of liquid parahydrogen at 14 K used in
  Fig 6; `data/Rabani_D_w_14K.csv` is the maximum-entropy reference spectrum.
- `spectra/*.npz` are precomputed model predictions (ensembles of 1000 realizations) so the
  figures can be reproduced without a GPU.

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the paper if you use this code.

## License and attribution

Released under the [MIT License](LICENSE). This work adapts
[`apapiu/transformer_latent_diffusion`](https://github.com/apapiu/transformer_latent_diffusion)
(MIT, © 2023 Alexandru Papiu); see [NOTICE](NOTICE).
