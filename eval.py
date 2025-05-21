#!/usr/bin/env python3
"""evaluate_beacon_net_multi.py – compare ≥ 1 checkpoints (scalar‑dB CNN **or**
sequence‑to‑DC LSTM/TorchScript).

* Accepts any mix of **state‑dict (.pt)** and **TorchScript (.pth)** files.
* If a model outputs a **(B,T,2)** complex sequence, we collapse it to a single
  complex gain with a mean over time and then convert to dBm so it is directly
  comparable to scalar‑dB models and to the FFT baseline.
* Unknown receptive‑field length can be forced with `--len-override N`.

Example
-------
python evaluate_beacon_net_multi.py \
       --data   ./dataset \
       --ckpts  best_beacon_cnn.pt  Deployment_len200_cpu.pth \
       --outdir eval_multi
"""

import argparse, csv, math, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.fft import fft

from model import BeaconPowerCNN   # scalar‑dB reference architecture

# ────────────────────────── helpers ─────────────────────────────────────
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
GAIN = 0.375   # Hann 3‑bin coherent gain

parse_levels = lambda name: (
    int(PAT_CW.search(name).group(1)),
    int(PAT_QPSK.search(name).group(1)),
    -int(PAT_QPSK.search(name).group(1)) - int(PAT_CW.search(name).group(1))
)

dBm_factor = 10.0 / math.log(10)

def gain_to_dBm(g: complex) -> float:  # |g|² reference 1 mW
    return dBm_factor * math.log((abs(g)**2) / 1e-3)

def fft_3bin_amp(x: np.ndarray) -> float:
    X = fft(x * np.hanning(len(x))) / len(x)
    return np.sqrt(np.sum(np.abs(X[[0, 1, -1]])**2) / GAIN)

def seq_to_gain(out_seq: torch.Tensor) -> torch.Tensor:
    """Collapse (B,T,2) decoded CW to a single complex gain."""
    re = out_seq[..., 0].mean(dim=1)
    im = out_seq[..., 1].mean(dim=1)
    return torch.complex(re, im)

# ────────────────────────── CLI ─────────────────────────────────────────
cli = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
cli.add_argument("--data",   required=True, type=Path)
cli.add_argument("--ckpts",  nargs="+", required=True, type=Path)
cli.add_argument("--outdir", default="eval_multi", type=Path)
cli.add_argument("--len-override", type=int, default=None,
                help="force input length for ALL models (useful for scripted")
args = cli.parse_args(); args.outdir.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cm     = plt.get_cmap("tab10")

# ────────────────────────── load models ────────────────────────────────
nets, labels, colors, in_len = [], [], [], []
for i, ck in enumerate(args.ckpts):
    expected = 1000  # default
    try:
        sd = torch.load(ck, map_location=DEVICE, weights_only=True)
        net = BeaconPowerCNN().to(DEVICE)
        net.load_state_dict(sd)
    except (TypeError, RuntimeError):
        net = torch.jit.load(ck, map_location=DEVICE)
        plist = list(net.parameters())
        if plist and plist[0].dim() == 3:
            k = plist[0].shape[-1]
            if k >= 50:  # treat tiny kernels as non‑informative
                expected = k
    net.eval(); nets.append(net)
    labels.append(ck.stem); colors.append(cm(i % 10)); in_len.append(expected)

if args.len_override:
    in_len = [args.len_override]*len(in_len)

# containers
by_sir = [defaultdict(list) for _ in nets]
by_sir_fft = defaultdict(list)

# ────────────────────────── loop over dataset ──────────────────────────
files = [p for fld in FOLDERS for p in (args.data/fld).glob("*.npz")]
for f in tqdm(files, unit="file"):
    _, _, sir = parse_levels(f.parent.name)
    with np.load(f, allow_pickle=True) as z:
        x_raw = z["x"].astype(np.complex64)
        meta   = z["meta"].item()
        rms    = np.sqrt(np.mean(np.abs(x_raw)**2))
        g_ref  = meta["pristine_gain"] / rms
    ref_dBm = gain_to_dBm(g_ref)

    x_n = x_raw / rms

    preds = []
    with torch.no_grad():
        for m, net in enumerate(nets):
            N = in_len[m]
            if N < len(x_n):
                s = (len(x_n)-N)//2; burst = x_n[s:s+N]
            elif N > len(x_n):
                p = (N-len(x_n))//2; burst = np.pad(x_n,(p,N-len(x_n)-p))
            else:
                burst = x_n
            # ---------- pack burst into tensor ----------
            if isinstance(net, torch.jit.ScriptModule):
                # Sequence models (e.g. LSTM) expect (B, T, 2) — channels‑last
                x_t = torch.tensor(
                    np.stack([burst.real, burst.imag], -1),  # (N,2)
                    dtype=torch.float32).unsqueeze(0).to(DEVICE)
            else:
                # Scalar‑dB CNN expects (B, 2, N) — channels‑first
                x_t = torch.tensor(
                    np.stack([burst.real, burst.imag], 0),   # (2,N)
                    dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # ---------- forward ----------
            out = net(x_t)
            if isinstance(out, tuple): out = out[0]  # (decoded, scale)
            if out.ndim == 3:  # sequence
                g_hat = seq_to_gain(out).cpu().item()
                pred = gain_to_dBm(g_hat)
            else:              # scalar dB
                pred = out.cpu().item()
            preds.append(pred)

    for m, p in enumerate(preds):
        by_sir[m][sir].append(p - ref_dBm)

    fft_dBm = 10*math.log10((fft_3bin_amp(x_n)**2)/1e-3)
    by_sir_fft[sir].append(fft_dBm - ref_dBm)

# ────────────────────────── plots ───────────────────────────────────────
sirs = np.array(sorted(by_sir[0]))
fig, ax = plt.subplots(figsize=(6.5,4))
for m,(col,lbl) in enumerate(zip(colors,labels)):
    means = [np.mean(by_sir[m][s]) for s in sirs]
    ax.plot(sirs,means,'o-',color=col,label=lbl)
ax.plot(sirs,[np.mean(by_sir_fft[s]) for s in sirs],'^--',color='k',label='FFT 3‑bin')
ax.set_xlabel('SIR (dB)'); ax.set_ylabel('Mean ΔPower (dB)'); ax.grid(ls=':')
ax.legend(frameon=False); fig.tight_layout()
fig.savefig(args.outdir/'mean_dA_vs_SIR.png',dpi=220)

# ────────────────────────── CSV ─────────────────────────────────────────
with (args.outdir / 'summary.csv').open('w', newline='') as fh:
    w = csv.writer(fh)
    header = ['SIR']
    for lbl in labels:
        header += [f'µ_{lbl}', f'σ_{lbl}']
    header += ['µ_FFT', 'σ_FFT']
    w.writerow(header)

    for s in sirs:
        row = [int(s)]
        # per‑model statistics
        for m in range(len(nets)):
            arr = np.asarray(by_sir[m][s])
            row += [f"{arr.mean():.3f}", f"{arr.std():.3f}"]
        # FFT baseline
        fft_arr = np.asarray(by_sir_fft[s])
        row += [f"{fft_arr.mean():.3f}", f"{fft_arr.std():.3f}"]
        w.writerow(row)

print("✓ Plots & CSV saved →", args.outdir.resolve())







