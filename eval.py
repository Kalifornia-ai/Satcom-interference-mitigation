#!/usr/bin/env python3
"""evaluate_beacon_net_multi.py – compare ≥ 1 checkpoints (scalar-dB CNN **or**
sequence-to-DC LSTM/TorchScript).

* Uses test_idx from the training JSON.
* Accepts state-dict (.pt) and TorchScript (.pth).
* Unifies all outputs to dBm so they’re directly comparable.
"""

from pathlib import Path
import argparse, csv, math, json, re
from collections import defaultdict

import numpy as np
import torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.fft import fft
from resmodel import HybridBeaconEstimator
from lstm import LSTMSeperatorSingle

from model import BeaconPowerCNN   # scalar-dB reference architecture
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
# ───────── load models (CNN • Hybrid • LSTM • scripted) ───────────────
nets, labels, colors, in_len = [], [], [], []

def identify_state_dict(sd):
    keys = list(sd)
    if "front.0.weight" in keys:                    # CNN
        return BeaconPowerCNN
    if "res.0.c1.weight" in keys:                   # Hybrid
        return HybridBeaconEstimator
    if "lstm.0.weight_ih_l0" in keys:               # LSTM
        return LSTMSeperatorSingle
    raise RuntimeError("Unknown state-dict format")

for i, ck in enumerate(args.ckpts):
    try:                                            # try plain state-dict
        sd = torch.load(ck, map_location=DEVICE, weights_only=True)
        NetClass = identify_state_dict(sd)
        net = NetClass().to(DEVICE)
        net.load_state_dict(sd)
        exp_len = 1000                              # all three take full burst
    except (RuntimeError, TypeError, FileNotFoundError):   # scripted
        net = torch.jit.load(ck, map_location=DEVICE)
        exp_len = 1000
        # crude receptive-field guess for very short scripted CNNs
        plist = list(net.parameters())
        if plist and plist[0].dim() == 3:
            k = plist[0].shape[-1]
            if k >= 50: exp_len = k
    net.eval()
    nets.append(net)
    labels.append(ck.stem)
    colors.append(cm(i % 10))
    in_len.append(exp_len)


# ────────────────────────── build test file list via meta JSON ─────────
meta_path = args.data / args.meta
if not meta_path.exists():
    raise SystemExit(f"JSON meta not found: {meta_path}")
meta = json.loads(meta_path.read_text())
try:
    test_idx = np.array(meta["test_idx"], dtype=int)
except KeyError:
    raise SystemExit("'test_idx' not found in JSON meta.")

files_all = np.array(sorted(args.data.rglob("*.npz")))
files     = files_all[test_idx]

print(f"Evaluating {len(files)} bursts …")

# ────────────────────────── containers ─────────────────────────────────
by_sir      = [defaultdict(list) for _ in nets]
by_sir_fft  = defaultdict(list)

# NEW: CW→QPSK buckets for box-plots
by_cw_qp     = [defaultdict(lambda: defaultdict(list)) for _ in nets]
by_cw_qp_fft = defaultdict(lambda: defaultdict(list))

# PSD bookkeeping
plotted_sir = set()

def db_spectrum(sig: np.ndarray):
    N = len(sig)
    S = np.fft.fftshift(np.fft.fft(sig*np.hanning(N))) / N
    f = np.fft.fftshift(np.fft.fftfreq(N, d=1/10e6)) / 1e6  # MHz
    P = 20*np.log10(np.abs(S) + 1e-18)
    return f, P

# ────────────────────────── main loop ──────────────────────────────────
for f in tqdm(files, unit="file"):
    cw_dBm, qpsk_dBm, sir = parse_levels(f.parent.name)
    with np.load(f, allow_pickle=True) as z:
        x_raw   = z["x"].astype(np.complex64)
        meta_npz= z["meta"].item()
        rms     = np.sqrt(np.mean(np.abs(x_raw)**2))
        g_ref   = meta_npz["pristine_gain"] / rms
    ref_dBm = gain_to_dBm(g_ref)
    x_n     = x_raw / rms

    # ---------- run all nets -------------------------------------------
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
            if isinstance(net, torch.jit.ScriptModule):
                x_t = torch.tensor(np.stack([burst.real, burst.imag], -1),
                                   dtype=torch.float32).unsqueeze(0).to(DEVICE)
            else:
                x_t = torch.tensor(np.stack([burst.real, burst.imag], 0),
                                   dtype=torch.float32).unsqueeze(0).to(DEVICE)
            out = net(x_t);  out = out[0] if isinstance(out, tuple) else out
            if isinstance(out, dict):                           # ← NEW
                out = out["gain"]                               # take the 2-vector tensor
            if out.ndim == 3:                               # (B,T,2) sequence
                g_hat = seq_to_gain(out).cpu().item()
                pred  = gain_to_dBm(g_hat)
            elif out.ndim == 2 and out.shape[-1] == 2:      # (B,2) re+im vector
                g_hat = complex(out[0,0].item(), out[0,1].item())
                pred  = gain_to_dBm(g_hat)
            else:                                           # scalar-dB output
                pred  = out.cpu().item()
            preds.append(pred)

    # ---------- PSD figures (once per SIR) ------------------------------
    if sir not in plotted_sir:
        f_MHz, P_mix = db_spectrum(x_n)
        y_gt = np.full_like(x_n, g_ref);  _, P_gt = db_spectrum(y_gt)
        # first net’s estimate
        mag_est = math.sqrt(1e-3 * 10**(preds[0]/10))
        y_hat_nn = np.full_like(x_n, mag_est*np.exp(1j*np.angle(g_ref)))
        _, P_hat_nn = db_spectrum(y_hat_nn)
        # FFT baseline estimate
        mag_fft = fft_3bin_amp(x_n)
        y_hat_fft = np.full_like(x_n, mag_fft*np.exp(1j*np.angle(g_ref)))
        _, P_hat_fft = db_spectrum(y_hat_fft)

        # full span
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(f_MHz,P_mix,lw=.7,label='mixture')
        ax.plot(f_MHz,P_gt,          label='ground truth')
        ax.plot(f_MHz,P_hat_nn,'--', label=f'estimate ({labels[0]})')
        ax.plot(f_MHz,P_hat_fft,':', label='FFT 3-bin')
        ax.set(xlabel='Freq (MHz)', ylabel='dB re RMS',
               xlim=(-5,5), ylim=(-150,0),
               title=f'PSD – SIR {sir} dB'); ax.grid(ls=':')
        ax.legend(frameon=False,ncol=4,fontsize=8)
        fig.tight_layout()
        fig.savefig(args.outdir/f'psd_SIR{sir}.png',dpi=220); plt.close(fig)

        # zoom ±100 kHz
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(f_MHz,P_mix,lw=.7); ax.plot(f_MHz,P_gt)
        ax.plot(f_MHz,P_hat_nn,'--'); ax.plot(f_MHz,P_hat_fft,':')
        ax.set(xlabel='Freq (MHz)', ylabel='dB re RMS',
               xlim=(-0.1,0.1), ylim=(-150,0),
               title=f'PSD zoom (±100 kHz) – SIR {sir} dB'); ax.grid(ls=':')
        ax.legend(['mixture','ground truth',
                   f'estimate ({labels[0]})','FFT 3-bin'],
                  frameon=False,ncol=4,fontsize=8)
        fig.tight_layout()
        fig.savefig(args.outdir/f'psd_zoom_SIR{sir}.png',dpi=220); plt.close(fig)
        plotted_sir.add(sir)

    # ---------- accumulate errors ---------------------------------------
    for m, p in enumerate(preds):
        err = p - ref_dBm
        by_sir[m][sir].append(err)
        by_cw_qp[m][cw_dBm][qpsk_dBm].append(err)

    fft_dBm  = 10*math.log10((fft_3bin_amp(x_n)**2)/1e-3)
    fft_err  = fft_dBm - ref_dBm
    by_sir_fft[sir].append(fft_err)
    by_cw_qp_fft[cw_dBm][qpsk_dBm].append(fft_err)

# ────────────────────────── aggregate plots & CSV ───────────────────────
sirs = np.array(sorted(by_sir[0]))

# 1) mean signed error vs SIR
fig, ax = plt.subplots(figsize=(6.5,4))
for m,(col,lbl) in enumerate(zip(colors,labels)):
    ax.plot(sirs,[np.mean(by_sir[m][s]) for s in sirs],'o-',color=col,label=lbl)
ax.plot(sirs,[np.mean(by_sir_fft[s]) for s in sirs],
        '^--',color='k',label='FFT 3-bin')
ax.set_xlabel('SIR (dB)'); ax.set_ylabel('Mean ΔPower (dB)')
ax.grid(ls=':'); ax.legend(frameon=False); fig.tight_layout()
fig.savefig(args.outdir/'mean_dA_vs_SIR.png',dpi=220); plt.close(fig)

# 2) absolute error vs SIR
fig, ax = plt.subplots(figsize=(6.5,4))
for m,(col,lbl) in enumerate(zip(colors,labels)):
    ax.plot(sirs,[np.mean(np.abs(by_sir[m][s])) for s in sirs],
            'o-',color=col,label=lbl)
ax.plot(sirs,[np.mean(np.abs(by_sir_fft[s])) for s in sirs],
        '^--',color='k',label='FFT 3-bin')
ax.set_xlabel('SIR (dB)'); ax.set_ylabel('Mean |ΔPower| (dB)')
ax.set_title('Absolute error versus SIR'); ax.grid(ls=':')
ax.legend(frameon=False); fig.tight_layout()
fig.savefig(args.outdir/'abs_dA_vs_SIR.png',dpi=220); plt.close(fig)

# 3) CSV summary
with (args.outdir/'summary.csv').open('w',newline='') as fh:
    wr = csv.writer(fh)
    hdr = ['SIR']+[f'µ_{l}' for l in labels]+['µ_FFT'] \
                 +[f'σ_{l}' for l in labels]+['σ_FFT']
    wr.writerow(hdr)
    for s in sirs:
        row = [s]
        for m in range(len(nets)):
            arr = np.asarray(by_sir[m][s]); row.append(f"{arr.mean():.3f}")
        fft_arr = np.asarray(by_sir_fft[s]); row.append(f"{fft_arr.mean():.3f}")
        for m in range(len(nets)):
            arr = np.asarray(by_sir[m][s]); row.append(f"{arr.std():.3f}")
        row.append(f"{fft_arr.std():.3f}")
        wr.writerow(row)

print("✓ Plots & CSV saved →", args.outdir.resolve())

# ------------------------------------------------------------------
# Box-plots: error vs QPSK level, one figure per CW tone
# ------------------------------------------------------------------
for cw in sorted(by_cw_qp[0]):                 # each CW tone
    qpsk_levels = sorted(by_cw_qp[0][cw])
    n_qp, n_nets = len(qpsk_levels), len(nets)
    data, pos, cols = [], [], []
    box_w = 0.8 / (n_nets+1)                  # +1 for FFT box

    for ix,q in enumerate(qpsk_levels):
        base = ix
        for m in range(n_nets):
            data.append(by_cw_qp[m][cw][q]);  pos.append(base-0.4+box_w/2+m*box_w)
            cols.append(colors[m])
        data.append(by_cw_qp_fft[cw][q]);     pos.append(base-0.4+box_w/2+n_nets*box_w)
        cols.append("k")

    fig, ax = plt.subplots(figsize=(7,4))
    bp = ax.boxplot(data, positions=pos, widths=box_w*0.9,
                    patch_artist=True, manage_ticks=False)
    for patch,c in zip(bp["boxes"],cols):
        patch.set_facecolor(c); patch.set_alpha(0.7)

    ax.set_xticks(range(n_qp))
    ax.set_xticklabels([f"{q} dBm" for q in qpsk_levels])
    ax.set_xlabel("QPSK power level")
    ax.set_ylabel("Carrier ΔPower error (dB)")
    ax.set_title(f"Error vs QPSK level   (CW = {cw} dBm)")
    ax.grid(ls=':')

    handles=[plt.Line2D([0],[0],color=colors[m],lw=6,label=labels[m])
             for m in range(n_nets)]
    handles.append(plt.Line2D([0],[0],color='k',lw=6,label='FFT 3-bin'))
    ax.legend(handles=handles,frameon=False)
    fig.tight_layout()
    fig.savefig(args.outdir/f'box_err_vs_QPSK_CW{cw}.png',dpi=220)
    plt.close(fig)

print("✓ Box-plot figures saved to", args.outdir.resolve())

