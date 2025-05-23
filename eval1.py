#!/usr/bin/env python3
"""
evaluate_beacon_net_multi.py – compare ≥1 checkpoints (CNN, Hybrid, LSTM, …).

* Reads test_idx from cw_power_train_meta.json.
* Accepts state-dict (.pt) and TorchScript (.pth/.pt) checkpoints.
* Detects per-checkpoint input layout (B,2,N) vs (B,N,2) on the fly.
* Normalises every burst to unit-RMS before feeding the network
  and scales the ground-truth gain in the same way.
* Converts all outputs to dBm so the metrics line up.
"""

from pathlib import Path
import argparse, csv, json, math, re
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.fft import fft

# YOUR three author models
from model   import BeaconPowerCNN
from resmodel import HybridBeaconEstimator
from lstm     import LSTMSeperatorSingle

# ────────── CLI ────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--data",   required=True, type=Path)
ap.add_argument("--ckpts",  nargs="+", required=True, type=Path)
ap.add_argument("--outdir", default="eval_multi", type=Path)
ap.add_argument("--meta",   default="cw_power_train_meta.json", type=Path)
ap.add_argument("--len-override", type=int, default=None)
args = ap.parse_args(); args.outdir.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cm = plt.get_cmap("tab10")
dBm_factor = 10.0 / math.log(10)
GAIN = 0.375

PAT_CW   = re.compile(r"Sine_50002[-_]?(-?\d+)dBm")
PAT_QPSK = re.compile(r"QPSK_5G[-_]?(-?\d+)dBm")
parse_levels = lambda p: (
    int(PAT_CW.search(p).group(1)),
    int(PAT_QPSK.search(p).group(1)),
    -int(PAT_QPSK.search(p).group(1)) - int(PAT_CW.search(p).group(1))
)

def gain_to_dBm(g): return dBm_factor * math.log((abs(g)**2) / 1e-3)

def seq_to_gain(t):  # (B,T,2) → complex
    return torch.complex(t[...,0].mean(1), t[...,1].mean(1))

def fft_3bin_amp(x):
    X = fft(x * np.hanning(len(x))) / len(x)
    return np.sqrt( (np.abs(X[[0,1,-1]])**2).sum() / GAIN )

# ────────── model loader ───────────────────────────────────────────────
def pick_class(sd):
    k = sd.keys()
    if "front.0.weight"      in k: return BeaconPowerCNN
    if "res.0.c1.weight"     in k: return HybridBeaconEstimator
    if "lstm.0.weight_ih_l0" in k: return LSTMSeperatorSingle
    raise RuntimeError("unknown state-dict")

nets, labels, colors, in_len, expect_time_major = [], [], [], [], []

for i, ck in enumerate(args.ckpts):
    try:                                # state-dict?
        sd  = torch.load(ck, map_location=DEVICE, weights_only=True)
        Net = pick_class(sd)
        net = Net().to(DEVICE); net.load_state_dict(sd)
        expect_time_major.append(False if Net is not LSTMSeperatorSingle else None)
        exp_len = 1000
    except (RuntimeError, TypeError, FileNotFoundError):
        net = torch.jit.load(ck, map_location=DEVICE)  # scripted
        expect_time_major.append(None)                 # let auto-probe decide
        exp_len = 1000
    nets.append(net.eval())
    labels.append(ck.stem);  colors.append(cm(i%10));  in_len.append(args.len_override or exp_len)

# ────────── dataset list ───────────────────────────────────────────────
meta = json.loads((args.data / args.meta).read_text())
test_idx = np.array(meta["test_idx"], dtype=int)
files = np.array(sorted(args.data.rglob("*.npz")))[test_idx]
print(f"Evaluating {len(files)} bursts …")

# ────────── accumulators ───────────────────────────────────────────────
by_sir = [defaultdict(list) for _ in nets];  by_sir_fft = defaultdict(list)
by_cw_qp = [defaultdict(lambda: defaultdict(list)) for _ in nets]
by_cw_qp_fft = defaultdict(lambda: defaultdict(list))

def build_tensor(burst, time_major):
    return torch.tensor(
        np.stack([burst.real, burst.imag], -1 if time_major else 0),
        dtype=torch.float32).unsqueeze(0).to(DEVICE)

# ────────── evaluation loop ────────────────────────────────────────────
for f in tqdm(files, unit="file"):
    cw, qpsk, sir = parse_levels(f.parent.name)
    with np.load(f, allow_pickle=True) as z:
        x_raw = z["x"].astype(np.complex64)
        g_raw = z["meta"].item()["pristine_gain"]

    rms  = np.sqrt(np.mean(np.abs(x_raw)**2))
    x_n  = x_raw / rms                         # unit-RMS burst
    g_ref= g_raw / rms                        # RMS-normalised gain
    ref_dBm = gain_to_dBm(g_ref)

    preds=[]
    for m,net in enumerate(nets):
        N=in_len[m]; burst=x_n
        if N<len(burst): s=(len(burst)-N)//2; burst=burst[s:s+N]
        elif N>len(burst): p=(N-len(burst))//2; burst=np.pad(burst,(p,N-len(burst)-p))

        ori = expect_time_major[m]
        for attempt in ([ori] if ori is not None else [False, True]):
            try:
                x_t = build_tensor(burst, attempt)
                out = net(x_t); out = out[0] if isinstance(out, tuple) else out
                if isinstance(out, dict): out = out["gain"]
                break
            except RuntimeError:
                if ori is not None: raise
                continue
        expect_time_major[m] = attempt        # remember good orientation

        if out.ndim==3:
            g_hat=seq_to_gain(out).cpu().item(); pred=gain_to_dBm(g_hat)
        elif out.ndim==2 and out.shape[-1]==2:
            g_hat=complex(out[0,0].item(), out[0,1].item()); pred=gain_to_dBm(g_hat)
        else:
            pred=out.cpu().item()
        preds.append(pred)

    for m,p in enumerate(preds):
        err=p-ref_dBm
        by_sir[m][sir].append(err)
        by_cw_qp[m][cw][qpsk].append(err)
    fft_dBm=10*math.log10((fft_3bin_amp(x_n)**2)/1e-3)
    by_sir_fft[sir].append(fft_dBm-ref_dBm)
    by_cw_qp_fft[cw][qpsk].append(fft_dBm-ref_dBm)

# ────────── quick mean-error plot ───────────────────────────────────────
sirs = np.array(sorted(by_sir[0]))
fig,ax=plt.subplots(figsize=(6,3))
for col,lbl,m in zip(colors,labels,range(len(nets))):
    ax.plot(sirs,[np.mean(np.abs(by_sir[m][s])) for s in sirs],'o-',color=col,label=lbl)
ax.plot(sirs,[np.mean(np.abs(by_sir_fft[s])) for s in sirs],'^--',color='k',label='FFT 3-bin')
ax.set_xlabel("SIR (dB)"); ax.set_ylabel("Mean ΔPower (dB)"); ax.grid(ls=':')
ax.legend(frameon=False); fig.tight_layout()
fig.savefig(args.outdir/"mean_dA_vs_SIR.png", dpi=220); plt.close(fig)

# ────────── CSV summary (µ / σ vs SIR) ─────────────────────────────────
with (args.outdir/'summary.csv').open('w',newline='') as fh:
    wr=csv.writer(fh)
    wr.writerow(['SIR']+[f'µ_{l}' for l in labels]+['µ_FFT']+
                        [f'σ_{l}' for l in labels]+['σ_FFT'])
    for s in sirs:
        row=[s]+[f"{np.mean(by_sir[m][s]):.3f}" for m in range(len(nets))]+\
            [f"{np.mean(by_sir_fft[s]):.3f}"]+\
            [f"{np.std(by_sir[m][s]):.3f}"  for m in range(len(nets))]+\
            [f"{np.std(by_sir_fft[s]):.3f}"]
        wr.writerow(row)

print("✓ evaluation complete – results saved to", args.outdir.resolve())

