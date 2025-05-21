#!/usr/bin/env python3
"""evaluate_beacon_net_multi.py – compare ≥1 BeaconPowerCNN or TorchScript checkpoints

Usage example
-------------
python evaluate_beacon_net_multi.py \
       --data   ./dataset \
       --ckpts  best_beacon_cnn.pt  DeploymentModel_len200_run1.pth \
       --outdir eval_multi

* Any mix of **state‑dict checkpoints** and **TorchScript archives** is
  accepted.  TorchScript models are detected automatically.
* If a model expects a shorter input (e.g. 200 samples), the script takes the
  **centre crop** of each 1 000‑sample burst to match its receptive field.
"""

import argparse, csv, math, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.fft import fft

from model import BeaconPowerCNN   # architecture for state‑dict checkpoints

# ────────────────────────── constants ────────────────────────────────────
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
PAT_CW   = re.compile(r"Sine_50002[-_]?(-?\d+)dBm")
PAT_QPSK = re.compile(r"QPSK_5G[-_]?(-?\d+)dBm")
GAIN = 0.375  # Hann 3‑bin coherent gain

dBm_factor = 10.0 / math.log(10)

def gain_to_dBm(g: complex) -> float:
    return dBm_factor * math.log((abs(g)**2) / 1e-3)

def fft_3bin_amp(x: np.ndarray) -> float:
    X = fft(x * np.hanning(len(x))) / len(x)
    return np.sqrt(np.sum(np.abs(X[[0, 1, -1]])**2) / GAIN)

parse_levels = lambda name: (
    int(PAT_CW.search(name).group(1)),
    int(PAT_QPSK.search(name).group(1)),
    -int(PAT_QPSK.search(name).group(1)) - int(PAT_CW.search(name).group(1))
)

# ────────────────────────── CLI ─────────────────────────────────────────
cli = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
cli.add_argument("--data",   required=True, type=Path)
cli.add_argument("--ckpts",  nargs="+",  required=True, type=Path,
                help="state‑dict (.pt) or TorchScript (.pth) files")
cli.add_argument("--outdir", default="eval_multi", type=Path)
args = cli.parse_args(); args.outdir.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cm     = plt.get_cmap("tab10")

# ────────────────────────── model loader ────────────────────────────────
nets, labels, colors, input_len = [], [], [], []
for i, ck in enumerate(args.ckpts):
    expected_len = 1000                # default (full burst)
    try:
        # -------- try as plain state‑dict (safe path) --------
        sd = torch.load(ck, map_location=DEVICE, weights_only=True)
        net = BeaconPowerCNN().to(DEVICE)
        net.load_state_dict(sd)
    except (TypeError, RuntimeError):   # -------- TorchScript fallback --------
        net = torch.jit.load(ck, map_location=DEVICE)
        # attempt to infer expected window length from first conv weight
        param_list = list(net.parameters())
        if param_list:                       # scripted module with params
            k = param_list[0]
            if k.dim() == 3:                 # (out,in,kernel)
                expected_len = int(k.shape[-1])
        # if no parameters, keep default 1000
    net.eval()
    nets.append(net)
    labels.append(ck.stem)
    colors.append(cm(i % 10))
    input_len.append(expected_len)

n_models = len(nets)

# containers
by_sir = [defaultdict(list) for _ in range(n_models)]
by_sir_fft = defaultdict(list)

# ────────────────────────── processing loop ────────────────────────────
files = [p for fld in FOLDERS for p in (args.data/fld).glob("*.npz")]
for f in tqdm(files, unit="file"):
    cw_dbm, _q_dbm, sir = parse_levels(f.parent.name)
    with np.load(f, allow_pickle=True) as z:
        x_raw = z["x"].astype(np.complex64)
        meta  = z["meta"].item()
        rms   = np.sqrt(np.mean(np.abs(x_raw)**2))
        g_ref = meta["pristine_gain"] / rms
    ref_dBm = gain_to_dBm(g_ref)

    x_n = x_raw / rms  # unit‑RMS

    # predictions for every model ---------------------------------------
    preds = []
    with torch.no_grad():
        for m, net in enumerate(nets):
            N = input_len[m]
            if N < len(x_n):                      # centre crop if needed
                start = (len(x_n) - N) // 2
                burst = x_n[start:start+N]
            elif N > len(x_n):                   # pad with zeros
                pad = (N - len(x_n))//2
                burst = np.pad(x_n, (pad, N-len(x_n)-pad))
            else:
                burst = x_n
            x_t = torch.tensor(np.stack([burst.real, burst.imag], 0)).unsqueeze(0).to(DEVICE)
            preds.append(net(x_t).item())

    for m, pred in enumerate(preds):
        by_sir[m][sir].append(pred - ref_dBm)

    # FFT baseline (always 1000 samples)
    fft_dBm = 10*math.log10((fft_3bin_amp(x_n)**2)/1e-3)
    by_sir_fft[sir].append(fft_dBm - ref_dBm)

# ────────────────────────── plots ───────────────────────────────────────
sirs = np.array(sorted(by_sir[0]))
fig, ax = plt.subplots(figsize=(6.5,4))
for m in range(n_models):
    means = [np.mean(by_sir[m][s]) for s in sirs]
    ax.plot(sirs, means, 'o-', color=colors[m], label=labels[m])
fft_means = [np.mean(by_sir_fft[s]) for s in sirs]
ax.plot(sirs, fft_means, '^--', color='k', label='FFT 3‑bin')
ax.set_xlabel('SIR (dB)'); ax.set_ylabel('Mean ΔPower (dB)'); ax.grid(ls=':')
ax.legend(frameon=False); fig.tight_layout()
fig.savefig(args.outdir/'mean_dA_vs_SIR.png', dpi=220)

# ────────────────────────── CSV ─────────────────────────────────────────
with (args.outdir/'summary.csv').open('w', newline='') as fh:
    w = csv.writer(fh)
    header = ['SIR']
    for lbl in labels: header += [f'µ_{lbl}', f'σ_{lbl}']
    header += ['µ_FFT', 'σ_FFT']; w.writerow(header)
    for s in sirs:
        row = [int(s)]
        for m in range(n_models):
            arr = np.asarray(by_sir[m][s]); row += [f"{arr.mean():.3f}", f"{arr.std():.3f}"]
        fft_arr = np.asarray(by_sir_fft[s]); row += [f"{fft_arr.mean():.3f}", f"{fft_arr.std():.3f}"]
        w.writerow(row)

print("✓ Plots & CSV saved ➜", args.outdir.resolve())



