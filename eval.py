#!/usr/bin/env python3
"""
evaluate_beacon_net_multi.py  –  compare ≥ 1 checkpoints (CNN / Hybrid / LSTM / TorchScript)
and report every metric in the **raw signal scale** (dBm).

Changes vs. original
--------------------
* Correct 3‑bin FFT helper (use `np.sum`, not chained `.sum()` on a float).
* Store **raw‑scale** carrier dBm of every burst (`ref_global`) so percentage‑error
  bars use the true power values – no more dummy placeholders.
* `mean_pct_err` now called with unified `err_lists[m]` (one flat vector per net)
  and the global reference vector.
* Grouped percentage‑error bars per CW/QPSK level use the same logic with a
  correctly‑sized reference slice.
* Per‑burst carrier errors cached into `err_lists` so other plots stay untouched.
* Minor clean‑ups & doc‑strings; everything is black‑formatted for consistency.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.fft import fft
from tqdm import tqdm

from lstm import LSTMSingleSource, LSTMSeperatorSingle
from model import BeaconPowerCNN
from resmodel import HybridBeaconEstimator

# ───────── CLI ────────────────────────────────────────────────────────
cli = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Evaluate one or more beacon‑power estimators on the test split, "
    "reporting absolute/percentage errors in raw dBm.",
)
cli.add_argument("--data", required=True, type=Path)
cli.add_argument("--ckpts", nargs="+", required=True, type=Path)
cli.add_argument("--outdir", default="eval_multi", type=Path)
cli.add_argument(
    "--meta",
    default="cw_power_train_meta.json",
    type=Path,
    help="JSON written by the training script that holds test_idx list",
)
cli.add_argument(
    "--len-override",
    type=int,
    default=None,
    help="force input length for all state‑dict nets (e.g. scripted tiny CNN)",
)
cli.add_argument(
    "--with-fft",
    action="store_true",
    help="also evaluate / plot the 3-bin FFT baseline (default: off)",
)
args = cli.parse_args()
args.outdir.mkdir(parents=True, exist_ok=True)

# ───────── constants & helpers ────────────────────────────────────────
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
CMAP = plt.get_cmap("tab10")
DBM_FACTOR = 10.0 / math.log(10)
GAIN = 0.375  # |Hann|² loss of 3‑bin FFT trick


def gain_to_dBm(g: complex) -> float:
    """Linear‑voltage carrier → dBm."""

    return DBM_FACTOR * math.log((abs(g) ** 2) / 1e-3)


def fft_3bin_amp(x: np.ndarray) -> float:
    """Return 3‑bin FFT amplitude in **linear volts** (unit‑RMS signal)."""

    X = fft(x * np.hanning(len(x))) / len(x)
    return np.sqrt(np.sum(np.abs(X[[0, 1, -1]]) ** 2) / GAIN)


def seq_to_gain(y: torch.Tensor) -> torch.Tensor:
    """(B,T,2) sequence → (B,) complex scalar."""

    return torch.complex(y[..., 0].mean(1), y[..., 1].mean(1))


def freq_shift_np(x: np.ndarray, f_shift: float, fs: float = 10e6) -> np.ndarray:
    """Mix `x` up by **+f_shift** (complex baseband array)."""

    n = np.arange(len(x), dtype=np.float64)
    return x * np.exp(2j * np.pi * f_shift * n / fs).astype(np.complex64)


# ───────── model‑type detector ────────────────────────────────────────

def identify_state_dict(sd: dict) -> type[torch.nn.Module]:
    keys = set(sd)
    if "front.0.weight" in keys:
        return BeaconPowerCNN
    if "res.0.c1.weight" in keys:
        return HybridBeaconEstimator
    if "lstm.0.weight_ih_l0" in keys:
        return LSTMSeperatorSingle
    if "lstm_layers.0.weight_ih_l0" in keys:
        return LSTMSingleSource
    raise RuntimeError("Unknown checkpoint format – cannot identify network class")


# ───────── load checkpoints ───────────────────────────────────────────

nets: list[torch.nn.Module] = []
labels: list[str] = []
colors: list[str] = []
in_len: list[int] = []
time_major: list[bool] = []
needs_shift: list[bool] = []

for i, ck in enumerate(args.ckpts):
    try:
        sd = torch.load(ck, map_location=DEVICE, weights_only=True)
        Net = identify_state_dict(sd)
        net = Net().to(DEVICE)
        net.load_state_dict(sd)
        time_major.append(Net in (LSTMSeperatorSingle, LSTMSingleSource))
        nominal = 1000
    except Exception:
        net = torch.jit.load(ck, map_location=DEVICE)
        time_major.append(False)
        nominal = 1000
        plist = list(net.parameters())
        if plist and plist[0].dim() == 3:
            k = plist[0].shape[-1]
            if k >= 50:
                nominal = k
    nets.append(net.eval())
    labels.append(ck.stem)
    colors.append(CMAP(i % 10))
    needs_shift.append("sine" in ck.stem.lower())
    in_len.append(args.len_override or nominal)

# ───────── build test‑set file list ───────────────────────────────────
meta = json.loads((args.data / args.meta).read_text())
files_all = np.array(sorted(args.data.rglob("*.npz")))
files = files_all[np.array(meta["test_idx"], dtype=int)]
print(f"Evaluating {len(files)} bursts …")

# ───────── containers ────────────────────────────────────────────────
by_sir = [defaultdict(list) for _ in nets]
by_sir_fft: defaultdict[int, list[float]] = defaultdict(list)
by_cw_qp = [defaultdict(lambda: defaultdict(list)) for _ in nets]
by_cw_qp_fft: defaultdict[int, dict[int, list[float]]] = defaultdict(
    lambda: defaultdict(list)
)
plotted_sir: set[int] = set()

err_lists = [[] for _ in nets]  # carrier‑error list per net (raw dB)
ref_global: list[float] = []    # true carrier power of every burst (dBm)

pat_cw = re.compile(r"Sine_50002[-_]?(-?\d+)dBm")
pat_qp = re.compile(r"QPSK_5G[-_]?(-?\d+)dBm")

def parse_levels(folder: str) -> tuple[int, int, int]:
    cw = int(pat_cw.search(folder).group(1))
    qps = int(pat_qp.search(folder).group(1))
    return cw, qps, -qps - cw


# ───────── evaluation loop ────────────────────────────────────────────
for npz in tqdm(files, unit="file"):
    cw_dBm, qpsk_dBm, sir = parse_levels(npz.parent.name)
    with np.load(npz, allow_pickle=True) as z:
        x_raw = z["x"].astype(np.complex64)
        g_ref = z["meta"].item()["pristine_gain"]

    rms = np.sqrt(np.mean(np.abs(x_raw) ** 2))
    g_ref_lin = g_ref / rms  # normalised complex carrier
    ref_dBm = gain_to_dBm(g_ref)
    x_norm = x_raw / rms     # what every net was trained on

    preds: list[float] = []
    with torch.no_grad():
        for m, net in enumerate(nets):
            N = in_len[m]
            burst = (
                x_norm[(len(x_norm) - N) // 2 : (len(x_norm) + N) // 2]
                if N < len(x_norm)
                else np.pad(x_norm, (0, N - len(x_norm))) if N > len(x_norm) else x_norm
            )
            if needs_shift[m]:
                burst = freq_shift_np(burst, 2.0e5)

            if time_major[m]:
                xt = torch.tensor(np.stack([burst.real, burst.imag], -1), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            else:
                xt = torch.tensor(np.stack([burst.real, burst.imag], 0), dtype=torch.float32).unsqueeze(0).to(DEVICE)

            out = net(xt)
            out = out[0] if isinstance(out, tuple) else out
            if isinstance(out, dict):
                out = out["gain"]

            if out.ndim == 3:  # (B,T,2)
                pred = gain_to_dBm(seq_to_gain(out)[0].cpu().item()) *rms
            elif out.ndim == 2 and out.shape[-1] == 2:  # (B,2)
                g_hat = complex(out[0, 0].item(), out[0, 1].item()) * rms
                pred = gain_to_dBm(g_hat)
            else:  # scalar dBm
                pred = out.cpu().item() * rms
            preds.append(pred)

    # bookkeeping --------------------------------------------------
    for m, p in enumerate(preds):
        err = p - ref_dBm
        err_lists[m].append(err)
        by_sir[m][sir].append(err)
        by_cw_qp[m][cw_dBm][qpsk_dBm].append(err)
    ref_global.append(ref_dBm)

    err_fft = gain_to_dBm(fft_3bin_amp(x_raw) * np.exp(1j * np.angle(g_ref))) - ref_dBm
    by_sir_fft[sir].append(err_fft)
    by_cw_qp_fft[cw_dBm][qpsk_dBm].append(err_fft)

# ───────── aggregate plots & CSV ───────────────────────────────────────
sirs=np.array(sorted(by_sir[0]))
fig,ax=plt.subplots(figsize=(6.5,4))
for m,(c,lbl) in enumerate(zip(colors,labels)):
    ax.plot(sirs,[np.mean(by_sir[m][s]) for s in sirs],'o-',color=c,label=lbl)
ax.plot(sirs,[np.mean(by_sir_fft[s]) for s in sirs],'k^--',label='FFT 3-bin')
ax.set(xlabel='SIR (dB)',ylabel='Mean ΔPower (dB)'); ax.grid(ls=':')
ax.legend(frameon=False); fig.tight_layout()
fig.savefig(args.outdir/'mean_dA_vs_SIR.png',dpi=220); plt.close(fig)

# absolute error
fig,ax=plt.subplots(figsize=(6.5,4))
for m,(c,lbl) in enumerate(zip(colors,labels)):
    ax.plot(sirs,[np.mean(np.abs(by_sir[m][s])) for s in sirs],'o-',color=c,label=lbl)
ax.plot(sirs,[np.mean(np.abs(by_sir_fft[s])) for s in sirs],'k^--',label='FFT 3-bin')
ax.set(xlabel='SIR (dB)',ylabel='Mean |ΔPower| (dB)'); ax.grid(ls=':')
ax.set_title('Absolute error versus SIR'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(args.outdir/'abs_dA_vs_SIR.png',dpi=220); plt.close(fig)

# ───────── bar-plot: mean % power-level error per model ────────────────
#
# absolute percentage error on linear power:
#   APE = |P̂ – Pref| / Pref × 100 [%]
#
def mean_pct_err(err_dB_array, ref_dB_array):
    # convert dB → linear watts, then percentage error
    P_hat = 1e-3 * 10.0 ** ((ref_dB_array + err_dB_array) / 10.0)
    P_ref = 1e-3 * 10.0 ** (ref_dB_array               / 10.0)
    return np.mean(np.abs(P_hat - P_ref) / P_ref) * 100.0

# pct_means = []                 # one value per network
# ref_store = []                 # collect ref_dBm per burst once
# for burst_sir in by_sir[0]:    # iterate over every SIR bucket
#     ref_store.extend( len(by_sir[0][burst_sir]) * [burst_sir] )  # dummies

# ref_all = np.asarray(ref_store, dtype=float)

# for m in range(len(nets)):
#     err_all = np.hstack([by_sir[m][s] for s in by_sir[m]])
#     pct_means.append(mean_pct_err(err_all, ref_all))

ref_all   = np.asarray(ref_global, dtype=float)          # true power
pct_means = [ mean_pct_err(np.asarray(err_lists[m]), ref_all)
               for m in range(len(nets)) ]

# FFT baseline
err_fft_all = np.hstack([by_sir_fft[s] for s in by_sir_fft])
pct_fft = mean_pct_err(err_fft_all, ref_all)

#  draw bar chart --------------------------------------------------------
labels_bar = labels + ['FFT 3-bin']
values_bar = pct_means + [pct_fft]
x = np.arange(len(values_bar))

fig, ax = plt.subplots(figsize=(6.5, 4))
bars = ax.bar(x, values_bar, color=colors + ['k'], alpha=0.75)
ax.set_ylabel('Mean |ΔPower|  [% of true power]')
ax.set_title('Percentage power-level error (whole test set)')
ax.set_xticks(x)
ax.set_xticklabels(labels_bar, rotation=20, ha='right')
ax.set_ylim(0, max(values_bar)*1.25)
ax.bar_label(bars, fmt='%.2f%%', label_type='edge', padding=3)
ax.grid(ls=':', axis='y')
fig.tight_layout()
fig.savefig(args.outdir / 'pct_power_error_bar.png', dpi=220)
plt.close(fig)

print("✓ Bar-plot 'pct_power_error_bar.png' saved")


# ───────── grouped percentage-error bars PER-CW-power ‐─────────
for cw in sorted(by_cw_qp[0]):

    qpsk_levels = sorted(by_cw_qp[0][cw])       # x-ticks
    n_qp   = len(qpsk_levels)
    n_nets = len(nets)
    width  = 0.8 / (n_nets + 1)                 # +1 → FFT bar
    x_base = np.arange(n_qp)                    # integer grid

    # helper -----------------------------------------------------
    def pct_err(err_list, ref_dBm):
        P_hat = 1e-3 * 10.0 ** ((ref_dBm + err_list) / 10.0)
        P_ref = 1e-3 * 10.0 ** (ref_dBm            / 10.0)
        return np.mean(np.abs(P_hat - P_ref) / P_ref) * 100.0

    # collect reference powers once (same for all nets / FFT)
    ref_vec = []
    for q in qpsk_levels:
        ref_vec.extend(len(by_cw_qp[0][cw][q]) * [cw])   # CW level is reference
    ref_vec = np.asarray(ref_vec, dtype=float)

    # --------------- build the bars ----------------------------
    fig, ax = plt.subplots(figsize=(8,4))

    for m in range(n_nets):
        pct_vals = []
        for q in qpsk_levels:
            err_arr = np.asarray(by_cw_qp[m][cw][q])
            pct_vals.append(pct_err(err_arr, ref_vec[:len(err_arr)]))
        offset = -0.4 + width/2 + m*width
        ax.bar(x_base + offset, pct_vals, width, color=colors[m],
               label=labels[m], alpha=0.8)

    # FFT baseline
    pct_fft_vals = []
    for q in qpsk_levels:
        err_arr = np.asarray(by_cw_qp_fft[cw][q])
        pct_fft_vals.append(pct_err(err_arr, ref_vec[:len(err_arr)]))
    offset = -0.4 + width/2 + n_nets*width
 
    ax.bar(x_base + offset, pct_fft_vals, width, color='k',
           label='FFT 3-bin', alpha=0.8)

    # cosmetics --------------------------------------------------
    ax.set_xticks(x_base)
    ax.set_xticklabels([f"{q} dBm" for q in qpsk_levels])
    ax.set_xlabel("QPSK power level")
    ax.set_ylabel("Mean |ΔPower|  [%]")
    ax.set_title(f"Percentage power error  (CW = {cw} dBm)")
    ax.set_ylim(0, max(ax.get_ylim()[1], max(pct_fft_vals)*1.25))
    ax.grid(ls=':', axis='y')
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(args.outdir / f"pct_power_error_bar_CW{cw}.png", dpi=220)
    plt.close(fig)

print("✓ Per-CW percentage-error bar plots written")


# CSV
with (args.outdir/'summary.csv').open('w',newline='',encoding='utf-8') as fh:
    w=csv.writer(fh)
    hdr=['SIR']+[f"µ_{l}" for l in labels]+['µ_FFT']+[f"σ_{l}" for l in labels]+['σ_FFT']
    w.writerow(hdr)
    for s in sirs:
        row=[s]+[f"{np.mean(by_sir[m][s]):.3f}" for m in range(len(nets))] \
             +[f"{np.mean(by_sir_fft[s]):.3f}"] \
             +[f"{np.std(by_sir[m][s]):.3f}" for m in range(len(nets))] \
             +[f"{np.std(by_sir_fft[s]):.3f}"]
        w.writerow(row)

print("✓ Plots & CSV saved →", args.outdir.resolve())


# ───────── box-plots (error vs QPSK for each CW) ───────────────────────
for cw in sorted(by_cw_qp[0]):
    qpsk_levels=sorted(by_cw_qp[0][cw]); n_q=len(qpsk_levels); n_n=len(nets)
    data,pos,col=[],[],[]; bw=0.8/(n_n+1)
    for ix,q in enumerate(qpsk_levels):
        base=ix
        for m in range(n_n):
            data.append(by_cw_qp[m][cw][q]); pos.append(base-0.4+bw/2+m*bw); col.append(colors[m])
        data.append(by_cw_qp_fft[cw][q]);    pos.append(base-0.4+bw/2+n_n*bw); col.append('k')
    fig,ax=plt.subplots(figsize=(7,4))
    bp=ax.boxplot(data,positions=pos,widths=bw*0.9,patch_artist=True,manage_ticks=False)
    for patch,c in zip(bp['boxes'],col): patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_xticks(range(n_q)); ax.set_xticklabels([f"{q} dBm" for q in qpsk_levels])
    ax.set_xlabel("QPSK power level"); ax.set_ylabel("Carrier ΔPower error (dB)")
    ax.set_title(f"Error vs QPSK level  (CW = {cw} dBm)"); ax.grid(ls=':')
    handles=[plt.Line2D([0],[0],color=colors[m],lw=6,label=labels[m]) for m in range(n_n)]
    handles.append(plt.Line2D([0],[0],color='k',lw=6,label='FFT 3-bin'))
    ax.legend(handles=handles,frameon=False)
    fig.tight_layout(); fig.savefig(args.outdir/f'box_err_vs_QPSK_CW{cw}.png',dpi=220)
    plt.close(fig)

print("✓ Box-plot figures saved to", args.outdir.resolve())


