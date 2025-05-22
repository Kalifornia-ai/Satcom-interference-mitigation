#!/usr/bin/env python3
"""evaluate_beacon_net_multi.py – compare ≥ 1 checkpoints (scalar‑dB CNN **or**
sequence‑to‑DC LSTM/TorchScript).

* Reads the **test_idx** list saved in the training JSON (default
  ``cw_power_train_meta.json``) instead of hard‑coding folder names.
* Accepts any mix of **state‑dict (.pt)** and **TorchScript (.pth)** files.
* If a model outputs a **(B,T,2)** complex sequence, we collapse it to a single
  complex gain and convert to dBm so it is directly comparable to scalar‑dB
  models and to the FFT baseline.
* Unknown receptive‑field length can be forced with `--len-override N`.

Example
-------
```bash
python evaluate_beacon_net_multi.py \
       --data   ./dataset \
       --ckpts  best_beacon_cnn.pt  deployment_len200_cpu.pth \
       --meta   cw_power_train_meta.json \
       --outdir eval_multi
```"""

from pathlib import Path
import argparse, csv, math, json, re
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.fft import fft

from model import BeaconPowerCNN   # scalar‑dB reference architecture

# ────────────────────────── CLI ─────────────────────────────────────────
cli = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
cli.add_argument("--data",   required=True, type=Path)
cli.add_argument("--ckpts",  nargs="+", required=True, type=Path)
cli.add_argument("--outdir", default="eval_multi", type=Path)
cli.add_argument("--meta",   default="cw_power_train_meta.json", type=Path,
                help="JSON file that holds test_idx list")
cli.add_argument("--len-override", type=int, default=None,
                help="force input length for ALL models (scripted ones in particular)")
args = cli.parse_args(); args.outdir.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cm     = plt.get_cmap("tab10")

dBm_factor = 10.0 / math.log(10)
GAIN = 0.375

PAT_CW   = re.compile(r"Sine_50002[-_]?(-?\d+)dBm")
PAT_QPSK = re.compile(r"QPSK_5G[-_]?(-?\d+)dBm")
parse_levels = lambda name: (
    int(PAT_CW.search(name).group(1)),
    int(PAT_QPSK.search(name).group(1)),
    -int(PAT_QPSK.search(name).group(1)) - int(PAT_CW.search(name).group(1))
)

def gain_to_dBm(g: complex) -> float:
    return dBm_factor * math.log((abs(g)**2) / 1e-3)

def fft_3bin_amp(x: np.ndarray) -> float:
    X = fft(x * np.hanning(len(x))) / len(x)
    return np.sqrt(np.sum(np.abs(X[[0, 1, -1]])**2) / GAIN)

def seq_to_gain(out_seq: torch.Tensor) -> torch.Tensor:
    re = out_seq[..., 0].mean(dim=1)
    im = out_seq[..., 1].mean(dim=1)
    return torch.complex(re, im)

# ────────────────────────── load models ────────────────────────────────
nets, labels, colors, in_len = [], [], [], []
for i, ck in enumerate(args.ckpts):
    expected = 1000
    try:
        sd = torch.load(ck, map_location=DEVICE, weights_only=True)
        net = BeaconPowerCNN().to(DEVICE)
        net.load_state_dict(sd)
    except (TypeError, RuntimeError):
        net = torch.jit.load(ck, map_location=DEVICE)
        plist = list(net.parameters())
        if plist and plist[0].dim() == 3:
            k = plist[0].shape[-1]
            if k >= 50:
                expected = k
    net.eval(); nets.append(net)
    labels.append(ck.stem); colors.append(cm(i % 10)); in_len.append(expected)

if args.len_override:
    in_len = [args.len_override]*len(in_len)

# ────────────────────────── build test file list via meta JSON ─────────
meta_path = args.data / args.meta
if not meta_path.exists():
    raise SystemExit(f"JSON meta not found: {meta_path}")
meta = json.loads(meta_path.read_text())
try:
    test_idx = np.array(meta["test_idx"], dtype=int)
except KeyError:
    raise SystemExit("'test_idx' not found in JSON meta. Re‑train with latest cw_power_model.py.")

# full sorted file list exactly like training loader does
files_all = np.array(sorted(args.data.rglob("*.npz")))
files = files_all[test_idx]

print(f"Evaluating {len(files)} bursts …")

# containers
by_sir = [defaultdict(list) for _ in nets]
by_sir_fft = defaultdict(list)

# ---------------------------------------------------------------------
# PSD helper + book-keeping
# ---------------------------------------------------------------------
plotted_sir = set()                     # keep track so we plot each SIR once

def db_spectrum(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hann-window PSD in dB re burst RMS (x is already unit-RMS)."""
    N   = len(sig)
    S   = np.fft.fftshift(np.fft.fft(sig*np.hanning(N))) / N
    f   = np.fft.fftshift(np.fft.fftfreq(N, d=1/10e6)) / 1e6   # MHz
    P_dB= 20*np.log10(np.abs(S) + 1e-18)                       # avoid log(0)
    return f, P_dB


for f in tqdm(files, unit="file"):
    _, _, sir = parse_levels(f.parent.name)
    with np.load(f, allow_pickle=True) as z:
        x_raw = z["x"].astype(np.complex64)
        meta_npz = z["meta"].item()
        rms = np.sqrt(np.mean(np.abs(x_raw)**2))
        g_ref = meta_npz["pristine_gain"] / rms
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
            # pack to tensor
            if isinstance(net, torch.jit.ScriptModule):
                x_t = torch.tensor(np.stack([burst.real, burst.imag], -1), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            else:
                x_t = torch.tensor(np.stack([burst.real, burst.imag], 0), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            out = net(x_t)
            if isinstance(out, tuple):
                out = out[0]
            if out.ndim == 3:
                g_hat = seq_to_gain(out).cpu().item()
                pred_dBm = gain_to_dBm(g_hat)
            else:
                pred_dBm = out.cpu().item()
            preds.append(pred_dBm)
    
   # --------------------------------------------------------------
# --- one PSD figure per SIR (mixture · ground-truth · estimate)
# --------------------------------------------------------------
if sir not in plotted_sir:
    # (a) mixture (already unit-RMS)
    f_MHz, P_mix = db_spectrum(x_n)

    # (b) ground-truth CW only
    y_gt = np.full_like(x_n, g_ref)
    _, P_gt = db_spectrum(y_gt)

    # (c) estimate from *first* model’s prediction
    pred_dBm = preds[0]
    mag_est  = math.sqrt(1e-3 * 10**(pred_dBm / 10))
    g_est    = mag_est * np.exp(1j * np.angle(g_ref))
    y_hat_nn = np.full_like(x_n, g_est)
    _, P_hat_nn = db_spectrum(y_hat_nn)

    # (d) estimate from FFT 3-bin baseline
    mag_fft  = fft_3bin_amp(x_n)          # linear volts
    g_fft    = mag_fft * np.exp(1j * np.angle(g_ref))
    y_hat_fft = np.full_like(x_n, g_fft)
    _, P_hat_fft = db_spectrum(y_hat_fft)

    # ---------- full-band PSD --------------------------------------------
    fig_psd, ax_psd = plt.subplots(figsize=(6, 3))
    ax_psd.plot(f_MHz, P_mix,       lw=.7, label='mixture')
    ax_psd.plot(f_MHz, P_gt,               label='ground truth')
    ax_psd.plot(f_MHz, P_hat_nn,    ls='--', label=f'estimate ({labels[0]})')
    ax_psd.plot(f_MHz, P_hat_fft,   ls=':',  label='FFT 3-bin')
    ax_psd.set(xlabel='Frequency (MHz)', ylabel='dB re RMS',
               title=f'PSD – SIR {sir} dB', xlim=(-5, 5))
    ax_psd.set_ylim(-150, 0); ax_psd.grid(ls=':')
    ax_psd.legend(frameon=False, ncol=4, fontsize=8)
    fig_psd.tight_layout()
    fig_psd.savefig(args.outdir / f'psd_SIR{sir}.png', dpi=220)
    plt.close(fig_psd)

    # ---------- zoomed PSD (±100 kHz) ------------------------------------
    fig_zoom, ax_zoom = plt.subplots(figsize=(6, 3))
    ax_zoom.plot(f_MHz, P_mix,       lw=.7)
    ax_zoom.plot(f_MHz, P_gt)
    ax_zoom.plot(f_MHz, P_hat_nn,    ls='--')
    ax_zoom.plot(f_MHz, P_hat_fft,   ls=':')
    ax_zoom.set(xlabel='Frequency (MHz)', ylabel='dB re RMS',
                title=f'PSD zoom (±100 kHz) – SIR {sir} dB',
                xlim=(-0.1, 0.1))                   # ±0.1 MHz = ±100 kHz
    ax_zoom.set_ylim(-150, 0); ax_zoom.grid(ls=':')
    ax_zoom.legend(['mixture', 'ground truth',
                    f'estimate ({labels[0]})', 'FFT 3-bin'],
                   frameon=False, ncol=4, fontsize=8)
    fig_zoom.tight_layout()
    fig_zoom.savefig(args.outdir / f'psd_zoom_SIR{sir}.png', dpi=220)
    plt.close(fig_zoom)

    plotted_sir.add(sir)



    for m, p in enumerate(preds):
        by_sir[m][sir].append(p - ref_dBm)

    fft_dBm = 10*math.log10((fft_3bin_amp(x_n)**2)/1e-3)
    by_sir_fft[sir].append(fft_dBm - ref_dBm)

# ────────────────────────── plots & CSV ────────────────────────────────

sirs = np.array(sorted(by_sir[0]))

# 1) Mean signed error vs SIR
fig, ax = plt.subplots(figsize=(6.5, 4))
for m, (col, lbl) in enumerate(zip(colors, labels)):
    means = [np.mean(by_sir[m][s]) for s in sirs]
    ax.plot(sirs, means, 'o-', color=col, label=lbl)
ax.plot(sirs, [np.mean(by_sir_fft[s]) for s in sirs], '^--', color='k', label='FFT 3-bin')
ax.set_xlabel('SIR (dB)'); ax.set_ylabel('Mean ΔPower (dB)'); ax.grid(ls=':')
ax.legend(frameon=False); fig.tight_layout()
fig.savefig(args.outdir / 'mean_dA_vs_SIR.png', dpi=220)
plt.close(fig)

# 2) Absolute error vs SIR
fig_abs, ax_abs = plt.subplots(figsize=(6.5, 4))
for m, (col, lbl) in enumerate(zip(colors, labels)):
    abs_means = [np.mean(np.abs(by_sir[m][s])) for s in sirs]
    ax_abs.plot(sirs, abs_means, 'o-', color=col, label=lbl)
abs_fft = [np.mean(np.abs(by_sir_fft[s])) for s in sirs]
ax_abs.plot(sirs, abs_fft, '^--', color='k', label='FFT 3-bin')
ax_abs.set_xlabel('SIR (dB)'); ax_abs.set_ylabel('Mean |ΔPower| (dB)')
ax_abs.set_title('Absolute error versus SIR')
ax_abs.grid(ls=':'); ax_abs.legend(frameon=False)
fig_abs.tight_layout(); fig_abs.savefig(args.outdir / 'abs_dA_vs_SIR.png', dpi=220)
plt.close(fig_abs)

# 3) CSV summary
csv_path = args.outdir / 'summary.csv'
with csv_path.open('w', newline='') as fh:
    writer = csv.writer(fh)
    header = ['SIR']
    for lbl in labels:
        header += [f'µ_{lbl}', f'σ_{lbl}']
    header += ['µ_FFT', 'σ_FFT']
    writer.writerow(header)

    for s in sirs:
        row = [int(s)]
        for m in range(len(nets)):
            arr = np.asarray(by_sir[m][s])
            row += [f"{arr.mean():.3f}", f"{arr.std():.3f}"]
        fft_arr = np.asarray(by_sir_fft[s])
        row += [f"{fft_arr.mean():.3f}", f"{fft_arr.std():.3f}"]
        writer.writerow(row)

print(f"✓ Plots & CSV saved → {args.outdir.resolve()}")
