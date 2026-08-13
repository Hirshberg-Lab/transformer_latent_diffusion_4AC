#!/usr/bin/env python3
"""
Regenerate the model predictions consumed by the figure scripts in ``create_figs/``.

This is a standalone, faithful port of the inference cells from the development
notebooks (``sharper_spectra.ipynb`` and ``deterministic_NN.ipynb``). It loads a
trained checkpoint and writes the ``spectra/*.npz`` bundles that the figures read.

    # Diffusion model -> spectra/data_with_diffusion.npz and spectra/diffusion_rabani_pred.npz
    python run_inference.py --model diffusion --model-name sharper_peaks_12layers

    # Regression baseline -> spectra/data_deterministic_<n>.npz
    python run_inference.py --model regression --model-number 1

Key settings (matching the paper, Sec. S2)
------------------------------------------
* Conditioning uses the four-channel G(tau) representation (``pointwise_norm=True``).
* Sampling: DPM-Solver++ (``use_ddpm_plus=True``), N=40 steps, pure conditional
  sampling (guidance scale 1.0, no kernel constraint).
* Uncertainty bounds use an ensemble of 1000 independent realizations per iTCF.

The precomputed outputs are already shipped in ``spectra/`` so the figures can be
reproduced without running this script. Run this only to regenerate them from a
trained model (requires the checkpoint and, in practice, a CUDA GPU).

Run this script from the repository root.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rnd_spectra.bumps import Bumps
from rnd_spectra.ontheflydataset import OnTheFlyDataset, pointwise_G_normalization
from tld.configs import SpectralDataConfig, DenoiserConfig, DeterministicConfig
from tld.diffusion import DiffusionGenerator

# --- Fixed settings ---------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
X_POINTS = 1024            # number of frequency points in C(omega)
N_TAU = 99                 # number of imaginary-time points in G(tau)
TAU = np.linspace(0, 1, N_TAU)

DENOISER_CONFIG = DenoiserConfig(n_layers=12, embed_dim=256)
DETERMINISTIC_CONFIG = DeterministicConfig(n_layers=12, embed_dim=256)

# Evaluation test set: 9 synthetic examples, fixed seed so every model is scored
# on the *same* iTCFs (this alignment is what the Fig. 3 comparison relies on).
TEST_DATA_CONFIG = SpectralDataConfig(
    omega_domain=(0, 50),
    num_bumps_range=(1, 4),
    bump_widths_fraction_range=(0.1, 0.45),
    bump_centers_fraction_range=(0.0, 0.6),
)
TEST_SEED = 40
TEST_SIZE = 9

# Global channel-wise normalization statistics for the four-channel G(tau)
# (identical to the OnTheFlyDataset defaults).
CHANNEL_MEAN = torch.Tensor([0.3319, -1.8394, 0.0023, 0.5002])
CHANNEL_STD = torch.Tensor([0.4170, 1.5327, 1.1582, 0.2884])


# --- Shared helpers ---------------------------------------------------------
def build_test_set():
    """Return (C_test, G_test) for the fixed 9-example synthetic evaluation set.

    C_test: (9, 1, 1024) ground-truth spectra.
    G_test: (9, 4, 99)   four-channel conditioning iTCFs.
    """
    bumps = Bumps(hparams=asdict(TEST_DATA_CONFIG), random_seed=TEST_SEED)
    dataset = OnTheFlyDataset(bumps=bumps, dataset_size=TEST_SIZE, pointwise_norm=True)
    loader = DataLoader(dataset, batch_size=TEST_SIZE, shuffle=False, num_workers=0)
    C_test, G_test = next(iter(loader))
    return C_test, G_test


def generate_ensemble(diffuser, labels, num_specs, n_realizations):
    """Draw ``n_realizations`` spectra per conditioning iTCF.

    Returns an array of shape (n_realizations, num_specs, 1024). Pure conditional
    sampling (class_guidance=1.0), DPM-Solver++ with 40 steps.
    """
    preds = []
    for seed in range(n_realizations):
        out = diffuser.generate(
            labels=labels,
            num_specs=num_specs,
            class_guidance=1.0,
            seed=seed,
            n_iter=40,
            exponent=1,
            x_points=X_POINTS,
            use_ddpm_plus=True,
        ).view(-1, X_POINTS).cpu().detach().numpy()
        preds.append(out)
    return np.array(preds)


def load_parahydrogen_inputs():
    """Load the two liquid-parahydrogen iTCFs used for Fig. 6 and interpolate them
    onto the model's tau grid. Returns 1-D numpy arrays (length 99), area-normalized.

    - Reference (Rabani et al.):  data/barak/G_Rabani_14.0K_180p.npy  (rows: tau, G)
    - PIMD (i-PI, 50 beads):      data/barak/G_PIMD_14.0K_180p.npy     (G values)
    """
    # Reference input.
    ref = np.load("data/barak/G_Rabani_14.0K_180p.npy")
    tau_ref, g_ref = ref[0], ref[1]
    g_ref = np.interp(TAU, tau_ref, g_ref)
    g_ref = g_ref / (np.trapz(g_ref, TAU) * np.pi)

    # PIMD input. The paper used an i-PI PIMD iTCF (50 beads); the in-repo copy is
    # data/barak/G_PIMD_14.0K_180p.npy (51 values on a 50-bead tau grid).
    g_pimd = np.load("data/barak/G_PIMD_14.0K_180p.npy")
    n_beads = len(g_pimd) - 1
    tau_pimd = np.linspace(0, 1, n_beads + 1)
    g_pimd = np.interp(TAU, tau_pimd, g_pimd)
    g_pimd = g_pimd / (np.trapz(g_pimd, TAU) * np.pi)

    return g_ref, g_pimd


# --- Diffusion inference ----------------------------------------------------
def load_diffusion_model(model_name, device=DEVICE):
    from tld.denoiser import Denoiser
    model = Denoiser(**asdict(DENOISER_CONFIG))
    state = torch.load(model_name, map_location=torch.device(device))
    model.load_state_dict(state["model_ema"])
    return model.to(device).eval()


def infer_diffusion(model, device=DEVICE, n_realizations=1000, out_dir="spectra"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diffuser = DiffusionGenerator(model.to(device), device, torch.float32)

    # 1) Synthetic evaluation set.
    C_test, G_test = build_test_set()
    print(f"[run_inference] diffusion: {n_realizations} realizations x {TEST_SIZE} synthetic iTCFs")
    C_preds = generate_ensemble(diffuser, G_test.to(device), TEST_SIZE, n_realizations)
    np.savez_compressed(
        out_dir / "data_with_diffusion.npz",
        C_preds=C_preds,
        G_test=G_test.cpu().numpy(),
        C_test=C_test.cpu().numpy(),
    )
    print(f"[run_inference] wrote {out_dir/'data_with_diffusion.npz'}  C_preds{C_preds.shape}")

    # 2) Liquid parahydrogen (Fig. 6).
    g_ref, g_pimd = load_parahydrogen_inputs()
    enc_G = pointwise_G_normalization(mean=CHANNEL_MEAN, std=CHANNEL_STD)
    print(f"[run_inference] diffusion: {n_realizations} realizations for parahydrogen iTCFs")
    labels_ref = enc_G(torch.Tensor(g_ref).view(1, -1)).unsqueeze(0).to(device)
    labels_pimd = enc_G(torch.Tensor(g_pimd).view(1, -1)).unsqueeze(0).to(device)
    C_ref = generate_ensemble(diffuser, labels_ref, 1, n_realizations)
    C_pimd = generate_ensemble(diffuser, labels_pimd, 1, n_realizations)
    np.savez_compressed(
        out_dir / "diffusion_rabani_pred.npz",
        C_Ref_input_preds=C_ref,
        C_PIMD_input_preds=C_pimd,
    )
    print(f"[run_inference] wrote {out_dir/'diffusion_rabani_pred.npz'}")


# --- Regression (baseline) inference ---------------------------------------
def load_regression_model(model_name, device=DEVICE):
    from tld.deterministic import Deterministic_NN
    model = Deterministic_NN(**asdict(DETERMINISTIC_CONFIG))
    state = torch.load(model_name, map_location=torch.device(device))
    model.load_state_dict(state["model_ema"])
    return model.to(device).eval()


def infer_regression(model, model_number, device=DEVICE, out_dir="spectra"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, G_test = build_test_set()
    with torch.no_grad():
        C_preds_det = model(G_test.to(device)).detach().cpu().numpy()
    out_path = out_dir / f"data_deterministic_{model_number}.npz"
    np.savez_compressed(out_path, C_preds_det=C_preds_det)
    print(f"[run_inference] wrote {out_path}  C_preds_det{C_preds_det.shape}")


# --- CLI --------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["diffusion", "regression"], default="diffusion")
    p.add_argument("--model-name", default=None,
                   help="Checkpoint file (diffusion default: sharper_peaks_12layers; "
                        "regression default: deterministic_model_<model-number>).")
    p.add_argument("--model-number", default="1",
                   help="Ensemble index for the regression baseline (default: 1).")
    p.add_argument("--n-realizations", type=int, default=1000,
                   help="Ensemble size per iTCF for the diffusion model (default: 1000).")
    return p


def main():
    args = build_parser().parse_args()
    if args.model == "diffusion":
        model_name = args.model_name or "sharper_peaks_12layers"
        model = load_diffusion_model(model_name)
        infer_diffusion(model, n_realizations=args.n_realizations)
    else:
        model_name = args.model_name or f"deterministic_model_{args.model_number}"
        model = load_regression_model(model_name)
        infer_regression(model, args.model_number)
    print("[run_inference] done.")


if __name__ == "__main__":
    main()
