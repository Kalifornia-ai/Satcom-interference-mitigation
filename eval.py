#!/usr/bin/env python3
"""evaluate_beacon_net.py  –  single‑pass evaluation of **BeaconPowerCNN**
=======================================================================
Compares the CNN’s CW‑power estimate against the pristine‑gain label and an
FFT 3‑bin baseline across all SIR folders, produces plots + CSV just like the
original hybrid script.

* Only **power error (dB)** is analysed – no phase/Δf metrics.
* Inputs are RMS‑normalised bursts (exactly as during training).
* Baseline FFT works on the **same** pre‑processed burst for a fair test.

Usage
-----
$ python evaluate_beacon_net.py --data ./dataset \
                               --ckpt best_beacon_cnn.pt \
                               --outdir eval_figs_power
"""

import argparse, csv, math, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.fft import fft
from scipy.stats import wilcoxon

from model import BeaconPowerCNN   # import your trained network

# ───────────────────────── configuration ────────────────────────────────
FOLDERS = [
    "Sine_50002-50dBm_QPSK_5G_-10dBm","Sine_50002-45dBm_QPSK_5G_-10dBm",
    "Sine_50002-40dBm_QPSK_5G_-10dBm","Sine_50002-45dBm_QPSK_5G_-20dBm",
    "Sine_50002-40dBm_QPSK_5G_-20dBm","Sine_50002-25dBm_QPSK_5G_-10dBm",
    "Sine_50002-20dBm_QPSK_5G_-10dBm","Sine_50002-25dBm_QPSK_5G_-20dBm",
    "Sine_50002-20dBm_QPSK_5G_-20dBm","Sine_50002-25dBm_QPSK_5G_-30dBm",
    "Sine_50002-20dBm_QPSK_5G_-30dBm","Sine_50002-25dBm_QPSK_5G_-40dBm",
    "Sine_50002-10dBm_QPSK_5G_-30dBm","Sine_50002-25dBm_QPSK_5G_-50dBm",
    "Sine_50002-10dBm_QPSK_5G_-40dBm","Sine_50002-10dBm_QPSK_5G_-50dBm",
]
PAT_CW, PAT_QPSK = re.compile(r"Sine_50002[-_]?(-?\d+)dBm"), re.compile(r"QPSK_5G[-_]?(-?\d+)dBm")
GAIN = 0.375                                     # Hann coherent gain (3‑bin)

# ───────────────────────── CLI ──────────────────────────────────────────
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--data",   required=True, type=Path)
ap.add_argument("--ckpt",   required=True, type=Path)
ap.add_argument("--outdir", default="eval_figs_power", type=Path)
ap.add_argument("--fs", type=float, default=10e6)
args = ap.parse_args(); args.outdir.mkdir(parents=True, exist_ok=True)

# ───────────────────────── model ────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
net = BeaconPowerCNN().to(DEVICE)
net.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
net.eval()

dBm_factor = 10.0 / math.log(10)

def gain_to_dBm(g: complex) -> float:
    return dBm_factor * math.log((abs(g)**2) / 1e-3)

# ───────────────────────── helpers ──────────────────────────────────────

def fft_3bin_amp(x: np.ndarray) -> float:
    X = fft(x * np.hanning(len(x))) / len(x)
    return np.sqrt(np.sum(np.abs(X[[0,1,-1]])**2) / GAIN)

# folder → (CW,pwr_QPSK, SIR)
parse_levels = lambda name: (
    int(PAT_CW.search(name).group(1)),
    int(PAT_QPSK.search(name).group(1)),
    -int(PAT_QPSK.search(name).group(1)) - int(PAT_CW.search(name).group(1))
)

# ───────────────────────── containers ───────────────────────────────────
by_sir, by_sir_fft = defaultdict(list), defaultdict(list)

# ───────────────────────── pass over dataset ────────────────────────────
files = [p for fld in FOLDERS for p in (args.data/fld).glob("*.npz")]
for f in tqdm(files, unit="file"):
    cw_dbm, _q_dbm, sir = parse_levels(f.parent.name)
    with np.load(f, allow_pickle=True) as z:
        x_raw = z["x"].astype(np.complex64)
        g_ref = z["meta"].item()["pristine_gain"]
    ref_dBm = gain_to_dBm(g_ref)

    # --- pre‑process like training ------------------------------------
    rms  = np.sqrt(np.mean(np.abs(x_raw)**2))
    x_n  = x_raw / rms
    x_t  = torch.tensor(np.stack([x_n.real, x_n.imag], 0)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_dBm = net(x_t).item()

    by_sir[sir].append(pred_dBm - ref_dBm)
    by_sir_fft[sir].append(20*math.log10(fft_3bin_amp(x_n)/1e-3) - ref_dBm)

# ───────────────────────── plotting & CSV (same as old) ────────────────
sirs = np.array(sorted(by_sir))
ml_means  = [np.mean(by_sir[s])     for s in sirs]
fft_means = [np.mean(by_sir_fft[s]) for s in sirs]

# --- simple scatter plot -------------------------------------------------
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(sirs, ml_means,  'o-', label='CNN')
ax.plot(sirs, fft_means, '^--', label='FFT 3‑bin')
ax.set_xlabel('SIR (dB)'); ax.set_ylabel('Mean ΔPower (dB)'); ax.grid(ls=':')
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(args.outdir/'mean_dA_vs_SIR.png', dpi=220)

# --- summary CSV & Wilcoxon ---------------------------------------------
csv_path = args.outdir/'summary.csv'
with open(csv_path, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['SIR','µ_CNN','σ_CNN','µ_FFT','σ_FFT','p_Wilcoxon'])
    for s in sirs:
        p = wilcoxon(by_sir_fft[s], by_sir[s]).pvalue
        w.writerow([int(s),
                    f"{np.mean(by_sir[s]):.3f}",  f"{np.std(by_sir[s]):.3f}",
                    f"{np.mean(by_sir_fft[s]):.3f}", f"{np.std(by_sir_fft[s]):.3f}",
                    f"{p:.2e}"])
print('✓ Figures & summary.csv written to', args.outdir.resolve())
