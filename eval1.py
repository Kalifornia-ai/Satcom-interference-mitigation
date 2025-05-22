#!/usr/bin/env python3
"""
evaluate_beacon_net_multi.py – compare ≥1 checkpoints (CNN, Hybrid, LSTM, …).

* Automatically detects input layout (B,2,N) vs (B,N,2) per checkpoint.
* Accepts state-dict (.pt) or TorchScript (.pth/.pt).
* Converts every output to dBm so all metrics are comparable.
"""

from pathlib import Path
import argparse, csv, json, math, re
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.fft import fft

from model   import BeaconPowerCNN
from resmodel import HybridBeaconEstimator
from lstm     import LSTMSeperatorSingle          # adjust import names if needed

# ─────────────── CLI ───────────────────────────────────────────────────
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--data",   required=True, type=Path)
ap.add_argument("--ckpts",  nargs="+",     required=True, type=Path)
ap.add_argument("--outdir", default="eval_multi", type=Path)
ap.add_argument("--meta",   default="cw_power_train_meta.json", type=Path)
ap.add_argument("--len-override", type=int, default=None)
args = ap.parse_args(); args.outdir.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dBm_factor = 10.0 / math.log(10)
GAIN = 0.375
cm = plt.get_cmap("tab10")

PAT_CW   = re.compile(r"Sine_50002[-_]?(-?\d+)dBm")
PAT_QPSK = re.compile(r"QPSK_5G[-_]?(-?\d+)dBm")
parse_levels = lambda p: (
    int(PAT_CW.search(p).group(1)),
    int(PAT_QPSK.search(p).group(1)),
    -int(PAT_QPSK.search(p).group(1))-int(PAT_CW.search(p).group(1))
)

def gain_to_dBm(g):  return dBm_factor*math.log((abs(g)**2)/1e-3)

def seq_to_gain(t):  # (B,T,2) → complex
    return torch.complex(t[...,0].mean(1), t[...,1].mean(1))

# ──────────── model loader ─────────────────────────────────────────────
def pick_class(sd):
    k = sd.keys()
    if "front.0.weight"      in k: return BeaconPowerCNN
    if "res.0.c1.weight"     in k: return HybridBeaconEstimator
    if "lstm.0.weight_ih_l0" in k: return LSTMSeperatorSingle
    raise RuntimeError("unknown state-dict")

nets, labels, colors, in_len = [], [], [], []
expect_time_major = []           # None = unknown  /  True / False

for i, ck in enumerate(args.ckpts):
    try:                                   # state-dict?
        sd = torch.load(ck, map_location=DEVICE, weights_only=True)
        net = pick_class(sd)().to(DEVICE); net.load_state_dict(sd)
        expect_time_major.append(False)    # all three author nets use (2,N)
    except (RuntimeError, TypeError, FileNotFoundError):
        net = torch.jit.load(ck, map_location=DEVICE)  # scripted
        expect_time_major.append(None)     # detect on first run
    net.eval()
    nets.append(net); labels.append(ck.stem)
    colors.append(cm(i%10)); in_len.append(args.len_override or 1000)

# ──────────── dataset list ─────────────────────────────────────────────
meta = json.loads((args.data/args.meta).read_text())
test_idx = np.array(meta["test_idx"], dtype=int)
files = np.array(sorted(args.data.rglob("*.npz")))[test_idx]
print(f"Evaluating {len(files)} bursts …")

# ──────────── accumulators ─────────────────────────────────────────────
by_sir = [defaultdict(list) for _ in nets]; by_sir_fft=defaultdict(list)
by_cw_qp=[defaultdict(lambda:defaultdict(list)) for _ in nets]
by_cw_qp_fft=defaultdict(lambda:defaultdict(list))

def fft_3bin_amp(x: np.ndarray) -> float:
    X = fft(x * np.hanning(len(x))) / len(x)
    return np.sqrt( (np.abs(X[[0, 1, -1]])**2).sum() / GAIN )

# ──────────── helper to build tensor in desired orientation ------------
def build_tensor(burst, time_major):
    if time_major:
        arr=np.stack([burst.real,burst.imag],-1)      # (N,2)
    else:
        arr=np.stack([burst.real,burst.imag],0)       # (2,N)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# ──────────── evaluation loop ─────────────────────────────────────────
for f in tqdm(files,unit="file"):
    cw,qpsk,sir = parse_levels(f.parent.name)
    with np.load(f,allow_pickle=True) as z:
        x_raw=z["x"].astype(np.complex64)
        g_ref=z["meta"].item()["pristine_gain"]
    ref_dBm=gain_to_dBm(g_ref)

    preds=[]
    for m,net in enumerate(nets):
        N=in_len[m]; burst=x_raw
        if N<len(burst): s=(len(burst)-N)//2; burst=burst[s:s+N]
        elif N>len(burst): p=(N-len(burst))//2; burst=np.pad(burst,(p,N-len(burst)-p))

        ori=expect_time_major[m] if expect_time_major[m] is not None else False
        for attempt in (ori, not ori):          # try current guess, then flip
            try:
                x_t=build_tensor(burst, attempt)
                out=net(x_t)
                out=out[0] if isinstance(out,tuple) else out
                if isinstance(out,dict): out=out["gain"]
                break
            except RuntimeError as e:
                if expect_time_major[m] is not None:
                    raise                    # orientation is fixed – real error
                ori=None                     # mark unknown until success
                continue
        # remember successful orientation
        if expect_time_major[m] is None: expect_time_major[m]=attempt

        # convert output to dBm
        if out.ndim==3:                       # (B,T,2)
            g_hat=seq_to_gain(out).cpu().item(); pred=gain_to_dBm(g_hat)
        elif out.ndim==2 and out.shape[-1]==2:
            g_hat=complex(out[0,0].item(),out[0,1].item()); pred=gain_to_dBm(g_hat)
        else: pred=out.cpu().item()
        preds.append(pred)

    for m,p in enumerate(preds):
        err=p-ref_dBm
        by_sir[m][sir].append(err)
        by_cw_qp[m][cw][qpsk].append(err)
    fft_dBm=10*math.log10((fft_3bin_amp(x_raw)**2)/1e-3)
    err=fft_dBm-ref_dBm
    by_sir_fft[sir].append(err); by_cw_qp_fft[cw][qpsk].append(err)

# ──────────── simple report (mean|abs| vs SIR) -------------------------
sirs=np.array(sorted(by_sir[0]))
fig,ax=plt.subplots(figsize=(6,3))
for col,lbl,m in zip(colors,labels,range(len(nets))):
    ax.plot(sirs,[np.mean(by_sir[m][s]) for s in sirs],'o-',color=col,label=lbl)
ax.plot(sirs,[np.mean(by_sir_fft[s]) for s in sirs],'^--',color='k',label='FFT')
ax.set_xlabel('SIR (dB)'); ax.set_ylabel('Mean ΔP (dB)'); ax.grid(ls=':')
ax.legend(frameon=False); fig.tight_layout()
fig.savefig(args.outdir/'mean_dA_vs_SIR.png',dpi=220); plt.close(fig)

print("✓ evaluation complete – results in", args.outdir.resolve())

