#!/usr/bin/env python3
"""
evaluate_beacon_net_multi.py  –  compare ≥ 1 checkpoints (CNN / Hybrid / LSTM /
TorchScript) on the **raw, un‑normalised scale**.
All reported metrics are dBm or % error referred to the true carrier power.
"""

from pathlib import Path
import argparse, csv, math, json, re
from collections import defaultdict
import numpy as np
import torch, matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.fft import fft

# local models ----------------------------------------------------------
from model   import BeaconPowerCNN
from resmodel import HybridBeaconEstimator
from lstm    import LSTMSeperatorSingle, LSTMSingleSource

# ─────────────── CLI ---------------------------------------------------
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--data",   required=True, type=Path)
ap.add_argument("--ckpts",  nargs="+",    required=True, type=Path)
ap.add_argument("--outdir", default="eval_multi", type=Path)
ap.add_argument("--meta",   default="cw_power_train_meta.json", type=Path)
ap.add_argument("--len-override", type=int, default=None,
                help="force input length for state‑dict models")
args = ap.parse_args(); args.outdir.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CMAP   = plt.get_cmap("tab10")
TEN_LN = 10.0 / math.log(10)
GAIN   = 0.375                          # |Hann|² loss of FFT 3‑bin

# ───────── helpers -----------------------------------------------------
def gain_to_dBm(g: complex) -> float:
    """Linear complex gain → dBm."""
    return TEN_LN * math.log((abs(g)**2) / 1e-3)

def fft_3bin_amp(x: np.ndarray) -> float:
    """3‑bin FFT amplitude estimate (linear volts)."""
    X = fft(x * np.hanning(len(x))) / len(x)
    return np.sqrt((np.abs(X[[0, 1, -1]])**2) .sum() / GAIN)

def seq_to_gain(y: torch.Tensor) -> torch.Tensor:
    """(B,T,2) → (B,) complex gain."""
    return torch.complex(y[..., 0].mean(1), y[..., 1].mean(1))

def freq_shift_np(x: np.ndarray, f_shift: float, fs: float = 10e6):
    """Frequency‑shift a complex burst by +f_shift Hz."""
    n = np.arange(len(x), dtype=np.float64)
    return (x * np.exp(2j * np.pi * f_shift * n / fs)).astype(np.complex64)

# ───────── identify model type from state‑dict ------------------------

def identify(sd):
    k = sd.keys()
    if "front.0.weight" in k:              return BeaconPowerCNN
    if "res.0.c1.weight" in k:             return HybridBeaconEstimator
    if "lstm.0.weight_ih_l0" in k:         return LSTMSeperatorSingle
    if "lstm_layers.0.weight_ih_l0" in k:  return LSTMSingleSource
    raise RuntimeError("Unknown checkpoint format → " + str(list(k)[:3]))

# ───────── load all checkpoints ---------------------------------------
nets, labels, colors, in_len, time_major, needs_shift = [], [], [], [], [], []
for i, ck in enumerate(args.ckpts):
    try:                                   # 1) raw state‑dict
        sd  = torch.load(ck, map_location=DEVICE, weights_only=True)
        cls = identify(sd); net = cls().to(DEVICE); net.load_state_dict(sd)
        time_major.append(cls in (LSTMSeperatorSingle, LSTMSingleSource))
    except Exception:                       # 2) TorchScript
        net = torch.jit.load(ck, map_location=DEVICE)
        time_major.append(False)
    nets.append(net.eval())
    labels.append(ck.stem)
    colors.append(CMAP(i % 10))
    needs_shift.append("sine" in ck.stem.lower())
    in_len.append(args.len_override or 1000)

# ───────── build test‑set list ----------------------------------------
meta = json.loads((args.data / args.meta).read_text())
test_idx = np.asarray(meta["test_idx"], dtype=int)
files = np.asarray(sorted(args.data.rglob("*.npz")))[test_idx]
print(f"Evaluating {len(files)} bursts …")

# ───────── containers --------------------------------------------------
by_sir       = [defaultdict(list) for _ in nets]
by_sir_fft   = defaultdict(list)
by_cw_qp     = [defaultdict(lambda: defaultdict(list)) for _ in nets]
by_cw_qp_fft = defaultdict(lambda: defaultdict(list))
plotted_sir  = set()
ref_powers_all = []            # carrier power per‑burst (dBm)

re_cw  = re.compile(r"Sine_50002[-_]?(-?\d+)dBm")
re_qps = re.compile(r"QPSK_5G[-_]?(-?\d+)dBm")
levels = lambda folder: (
    int(re_cw .search(folder).group(1)),
    int(re_qps.search(folder).group(1)),
    -int(re_qps.search(folder).group(1)) - int(re_cw.search(folder).group(1))
)

# PSD plotting helper ---------------------------------------------------

def db_spectrum(sig):
    N = len(sig)
    S = np.fft.fftshift(np.fft.fft(sig * np.hanning(N))) / N
    f = np.fft.fftshift(np.fft.fftfreq(N, d=1 / 10e6)) / 1e6
    return f, 20 * np.log10(np.abs(S) + 1e-18)

# ───────── main loop ---------------------------------------------------
for npz in tqdm(files, unit="file"):
    cw_dBm, qpsk_dBm, sir = levels(npz.parent.name)
    with np.load(npz, allow_pickle=True) as z:
        x_raw = z["x"].astype(np.complex64)
        g_ref = z["meta"].item()["pristine_gain"]   # raw‑scale complex gain
    ref_dBm = gain_to_dBm(g_ref)
    ref_powers_all.append(ref_dBm)

    rms = np.sqrt(np.mean(np.abs(x_raw)**2))      # magnitude for nets
    x_norm = x_raw / rms

    preds = []
    with torch.no_grad():
        for m, net in enumerate(nets):
            burst = x_norm.copy()
            N = in_len[m]
            if N < len(burst):  s = (len(burst) - N) // 2; burst = burst[s:s+N]
            elif N > len(burst): p = (N - len(burst)) // 2; burst = np.pad(burst, (p, N - len(burst) - p))
            if needs_shift[m]:
                burst = freq_shift_np(burst, 2.0e5) / rms   # shift still unit‑RMS

            if time_major[m]:
                xt = torch.tensor(np.stack([burst.real, burst.imag], -1), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            else:
                xt = torch.tensor(np.stack([burst.real, burst.imag], 0), dtype=torch.float32).unsqueeze(0).to(DEVICE)

            out = net(xt); out = out[0] if isinstance(out, tuple) else out
            if isinstance(out, dict): out = out["gain"]

            # convert to raw‑scale dBm
            if out.ndim == 3:                # sequence Re/Im
                g_hat = seq_to_gain(out)[0].cpu().item() * rms
                pred  = gain_to_dBm(g_hat)
            elif out.ndim == 2 and out.shape[-1] == 2:  # single Re/Im
                g_hat = complex(out[0,0].item(), out[0,1].item()) * rms
                pred  = gain_to_dBm(g_hat)
            else:                              # scalar dBm predicted on norm.
                pred  = out.cpu().item() + 20 * math.log10(rms)
            preds.append(pred)

    # baseline FFT on *raw* burst
    fft_est = fft_3bin_amp(x_raw) * np.exp(1j * np.angle(g_ref))
    fft_err = gain_to_dBm(fft_est) - ref_dBm
    by_sir_fft[sir].append(fft_err)
    by_cw_qp_fft[cw_dBm][qpsk_dBm].append(fft_err)

    # bucket errors per network
    for m, p_dBm in enumerate(preds):
        err = p_dBm - ref_dBm
        by_sir[m][sir].append(err)
        by_cw_qp[m][cw_dBm][qpsk_dBm].append(err)

    # PSD figure (once per SIR) ----------------------------------------
    if sir not in plotted_sir:
        f_MHz, P_mix = db_spectrum(x_raw)
        y_gt  = np.full_like(x_raw, g_ref); _, P_gt  = db_spectrum(y_gt)
        y_hat = np.full_like(x_raw, math.sqrt(1e-3 * 10**(preds[0] / 10)) * np.exp(1j * np.angle(g_ref)))
        _, P_hat = db_spectrum(y_hat)
        y_fft = np.full_like(x_raw, fft_est); _, P_fft = db_spectrum(y_fft)
        for rng, tag in [((-5, 5), 'full'), ((-0.1, 0.1), 'zoom')]:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(f_MHz, P_mix, lw=.7, label='mixture')
            ax.plot(f_MHz, P_gt, label='ground truth')
            ax.plot(f_MHz, P_hat, '--', label=f'estimate ({labels[0]})')
            ax.plot(f_MHz, P_fft, ':', label='FFT 3-bin')
            ax.set(xlabel='Freq (MHz)', ylabel='dB re 1 mW', xlim=rng, ylim=(-150, 0),
                   title=f'PSD {tag} – SIR {sir} dB')
            ax.grid(ls=':'); ax.legend(frameon=False, ncol=4, fontsize=8)
            fig.tight_layout(); fig.savefig(args.outdir / f'psd_{tag}_SIR{sir}.png', dpi=220)
            plt.close(fig)
        plotted_sir.add(sir)

# ───────── aggregate metrics & plots ----------------------------------
sirs = np.asarray(sorted(by_sir[0]))

# mean signed error vs SIR
fig, ax = plt.subplots(figsize=(6.5, 4))
for m, (c, lbl) in enumerate(zip(colors, labels)):
    ax.plot(sirs, [np.mean(by_sir[m][s]) for s in sirs], 'o-', color=c, label=lbl)
ax.plot(sirs, [np.mean(by_sir_fft[s]) for s in sirs], 'k^--', label='FFT 3-bin')
ax.set(xlabel='SIR (dB)', ylabel='Mean ΔPower (dB)'); ax.grid(ls=':')
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(args.outdir / 'mean_dA_vs_SIR.png', dpi=220)
plt.close(fig)

# mean |error| vs SIR
fig, ax = plt.subplots(figsize=(6.5, 4))
for m, (c, lbl) in enumerate(zip(colors, labels)):
    ax.plot(sirs, [np.mean(np.abs(by_sir[m][s])) for s in sirs], 'o-', color=c, label=lbl)
ax.plot(sirs, [np.mean(np.abs(by_sir_fft[s])) for s in sirs], 'k^--', label='FFT 3-bin')
ax.set(xlabel='SIR (dB)', ylabel='Mean |ΔPower| (dB)'); ax.grid(ls=':')
ax.set_title('Absolute error versus SIR'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(args.outdir / 'abs_dA_vs_SIR.png', dpi=220); plt.close(fig)

# percentage‑error bar (whole test set)

def pct_err(err, ref):
    P_hat = 1e-3 * 10 ** ((ref + err) / 10.0)
    P_ref = 1e-3 * 10 ** (ref / 10.0)
    return np.abs(P_hat - P_ref) / P_ref * 100.0

pct_means = []
ref_all = np.asarray(ref_powers_all, dtype=float)
for m in range(len(nets)):
    err_all = np.hstack([by_sir[m][s] for s in by_sir[m]])
    pct_means.append(pct_err(err_all, ref_all).mean())
err_fft_all = np.hstack([by_sir_fft[s] for s in by_sir_fft])
pct_fft = pct_err(err_fft_all, ref_all).mean()

labels_bar = labels + ['FFT 3-bin']
values_bar = pct_means + [pct_fft]

x = np.arange(len(values_bar))
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.bar(x, values_bar, color=colors + ['k'], alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels_bar, rotation=20, ha='right')
ax.set_ylabel('Mean |ΔPower| [%]'); ax.set_title('Percentage power error (all test)')
ax.grid(ls=':', axis='y')
fig.tight_layout(); fig.savefig(args.outdir / 'pct_power_error_bar.png', dpi=220); plt.close(fig)

# per‑CW percentage‑error grouped bars ---------------------------------
for cw in sorted(by_cw_qp[0]):
    qpsk_levels = sorted(by_cw_qp[0][cw]); n_q = len(qpsk_levels); n_n = len(nets)
    x_base = np.arange(n_q); bw = 0.8 / (n_n + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for m in range(n_n):
        pct_vals = []
        for q in qpsk_levels:
            err_arr = np.asarray(by_cw_qp[m][cw][q])
            ref_arr = np.full_like(err_arr, cw, dtype=float)
            pct_vals.append(pct_err(err_arr, ref_arr).mean())
        offset = -0.4 + bw/2 + m*bw
        ax.bar(x_base + offset, pct_vals, bw, color=colors[m], label=labels[m], alpha=0.8)
    pct_vals_fft = []
    for q in qpsk_levels:
        err_arr = np.asarray(by_cw_qp_fft[cw][q])
        ref_arr = np.full_like(err_arr, cw, dtype=float)
        pct_vals_fft.append(pct_err(err_arr, ref_arr).mean())
    ax.bar(x_base - 0.4 + bw/2 + n_n*bw, pct_vals_fft, bw, color='k', label='FFT 3-bin', alpha=0.8)
    ax.set_xticks(x_base); ax.set_xticklabels([f"{q} dBm" for q in qpsk_levels])
    ax.set_xlabel('QPSK power level'); ax.set_ylabel('Mean |ΔPower| [%]')
    ax.set_title(f'Percentage power error (CW = {cw} dBm)'); ax.grid(ls=':', axis='y')
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(args.outdir / f'pct_power_error_bar_CW{cw}.png', dpi=220); plt.close(fig)

print("✓ All plots & CSV written to", args.outdir.resolve())



