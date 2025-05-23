#!/usr/bin/env python3
"""
evaluate_beacon_net_multi.py – compare ≥ 1 checkpoints (CNN / Hybrid / LSTM /
TorchScript).  All outputs converted to dBm so metrics are comparable.
"""

from pathlib import Path
import argparse, csv, math, json, re
from collections import defaultdict
import numpy as np
import torch, matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.fft import fft
from model   import BeaconPowerCNN
from resmodel import HybridBeaconEstimator
from lstm    import LSTMSeperatorSingle, LSTMSingleSource      # ← two variants

# ───────── CLI ─────────────────────────────────────────────────────────
p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--data",   required=True, type=Path)
p.add_argument("--ckpts",  nargs="+",    required=True, type=Path)
p.add_argument("--outdir", default="eval_multi", type=Path)
p.add_argument("--meta",   default="cw_power_train_meta.json", type=Path)
p.add_argument("--len-override", type=int, default=None,
               help="force input length for all *state-dict* models")
args = p.parse_args(); args.outdir.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CMAP   = plt.get_cmap("tab10")
dBm_fac= 10.0 / math.log(10)
GAIN   = 0.375                    # |Hann|² power loss of three-bin FFT trick

def gain_to_dBm(g: complex) -> float:
    return dBm_fac * math.log((abs(g)**2)/1e-3)

def fft_3bin_amp(x: np.ndarray) -> float:             # linear volts (unit-RMS)
    X = fft(x*np.hanning(len(x))) / len(x)
    return np.sqrt(np.sum(np.abs(X[[0,1,-1]])**2) / GAIN)

def seq_to_gain(y: torch.Tensor) -> torch.Tensor:     # (B,T,2) → (B,) complex
    return torch.complex(y[...,0].mean(1), y[...,1].mean(1))

# ───────── helper: recognise a state-dict ──────────────────────────────
def identify(sd):
    k = list(sd)
    if "front.0.weight"   in k: return BeaconPowerCNN
    if "res.0.c1.weight"  in k: return HybridBeaconEstimator
    if "lstm.0.weight_ih_l0" in k:     return LSTMSeperatorSingle
    if "lstm_layers.0.weight_ih_l0" in k: return LSTMSingleSource
    raise RuntimeError("state-dict structure not recognised")

# ───────── Load all checkpoints ────────────────────────────────────────
nets, labels, colors, in_len, time_major = [], [], [], [], []
for i, ck in enumerate(args.ckpts):
    try:                                  # 1) raw state-dict
        sd  = torch.load(ck, map_location=DEVICE, weights_only=True)
        cls = identify(sd)
        net = cls().to(DEVICE); net.load_state_dict(sd)
        time_major.append(cls in (LSTMSeperatorSingle, LSTMSingleSource))
        nominal_len = 1000                # all training bursts are 1 k
    except (FileNotFoundError, RuntimeError, TypeError):   # 2) TorchScript
        net = torch.jit.load(ck, map_location=DEVICE)
        time_major.append(False)
        nominal_len = 1000
        if list(net.parameters()) and list(net.parameters())[0].dim()==3:
            k = list(net.parameters())[0].shape[-1]
            if k >= 50: nominal_len = k   # very tiny scripted CNN
    nets   .append(net.eval())
    labels .append(ck.stem)
    colors .append(CMAP(i%10))
    in_len .append(args.len_override or nominal_len)

# ───────── build test-file list ----------------------------------------
meta = json.loads((args.data/args.meta).read_text())
test_idx = np.array(meta["test_idx"],dtype=int)
files = np.array(sorted(args.data.rglob("*.npz")))[test_idx]
print(f"Evaluating {len(files)} bursts …")

# ───────── containers ---------------------------------------------------
by_sir       = [defaultdict(list) for _ in nets]
by_sir_fft   = defaultdict(list)
by_cw_qp     = [defaultdict(lambda: defaultdict(list)) for _ in nets]
by_cw_qp_fft = defaultdict(lambda: defaultdict(list))
plotted_sir  = set()

pat_cw = re.compile(r"Sine_50002[-_]?(-?\d+)dBm")
pat_qp = re.compile(r"QPSK_5G[-_]?(-?\d+)dBm")
def levels(folder):
    cw  = int(pat_cw.search(folder).group(1))
    qps = int(pat_qp.search(folder).group(1))
    return cw, qps, -qps-cw

def db_spectrum(sig):
    N=len(sig); S=np.fft.fftshift(np.fft.fft(sig*np.hanning(N)))/N
    f=np.fft.fftshift(np.fft.fftfreq(N,d=1/10e6))/1e6
    return f, 20*np.log10(np.abs(S)+1e-18)

# ───────── main evaluation loop ─────────────────────────────────────────
for npz in tqdm(files, unit="file"):
    cw_dBm,qpsk_dBm,sir = levels(npz.parent.name)
    with np.load(npz,allow_pickle=True) as z:
        x_raw = z["x"].astype(np.complex64)
        g_ref = z["meta"].item()["pristine_gain"]
    rms = np.sqrt(np.mean(np.abs(x_raw)**2))
    g_ref /= rms; ref_dBm = gain_to_dBm(g_ref)
    x_n = x_raw / rms

    preds=[]
    with torch.no_grad():
        for m,net in enumerate(nets):
            N=in_len[m]
            # centre-crop / pad burst to expected length
            if N<len(x_n):  s=(len(x_n)-N)//2; burst=x_n[s:s+N]
            elif N>len(x_n):p=(N-len(x_n))//2; burst=np.pad(x_n,(p,N-len(x_n)-p))
            else:           burst=x_n
            if time_major[m]:
                xt=torch.tensor(np.stack([burst.real,burst.imag],-1),
                                dtype=torch.float32).unsqueeze(0).to(DEVICE)
            else:
                xt=torch.tensor(np.stack([burst.real,burst.imag],0),
                                dtype=torch.float32).unsqueeze(0).to(DEVICE)
            out=net(xt);  out = out[0] if isinstance(out,tuple) else out
            if isinstance(out,dict): out=out["gain"]
            if out.ndim==3:                    # sequence Re/Im
                pred = gain_to_dBm(seq_to_gain(out)[0].cpu().numpy())
            elif out.ndim==2 and out.shape[-1]==2:   # single Re/Im
                g=complex(out[0,0].item(),out[0,1].item()); pred=gain_to_dBm(g)
            else:                                   # scalar dBm
                pred=out.cpu().item()
            preds.append(pred)

    #  PSD picture (once per SIR) -----------------------
    if sir not in plotted_sir:
        f_MHz,P_mix=db_spectrum(x_n)
        y_gt=np.full_like(x_n,g_ref); _,P_gt=db_spectrum(y_gt)
        y_hat=np.full_like(x_n,
              math.sqrt(1e-3*10**(preds[0]/10))*np.exp(1j*np.angle(g_ref)))
        _,P_hat=db_spectrum(y_hat)
        y_fft=np.full_like(x_n,
              fft_3bin_amp(x_n)*np.exp(1j*np.angle(g_ref)))
        _,P_fft=db_spectrum(y_fft)

        for rng,tag in [((-5,5),'full'),((-0.1,0.1),'zoom')]:
            fig,ax=plt.subplots(figsize=(6,3))
            ax.plot(f_MHz,P_mix,lw=.7,label='mixture')
            ax.plot(f_MHz,P_gt,         label='ground truth')
            ax.plot(f_MHz,P_hat,'--',   label=f'estimate ({labels[0]})')
            ax.plot(f_MHz,P_fft,':',    label='FFT 3-bin')
            ax.set(xlabel='Freq (MHz)',ylabel='dB re RMS',
                   xlim=rng,ylim=(-150,0),
                   title=f'PSD {tag} – SIR {sir} dB')
            ax.grid(ls=':'); ax.legend(frameon=False,ncol=4,fontsize=8)
            fig.tight_layout()
            fig.savefig(args.outdir/f'psd_{tag}_SIR{sir}.png',dpi=220)
            plt.close(fig)
        plotted_sir.add(sir)

    #  stats -------------------------------------------
    for m,p in enumerate(preds):
        err=p-ref_dBm
        by_sir[m][sir].append(err)
        by_cw_qp[m][cw_dBm][qpsk_dBm].append(err)
    fft_err=gain_to_dBm(fft_3bin_amp(x_n)*np.exp(1j*np.angle(g_ref)))-ref_dBm
    by_sir_fft[sir].append(fft_err)
    by_cw_qp_fft[cw_dBm][qpsk_dBm].append(fft_err)

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

pct_means = []                 # one value per network
ref_store = []                 # collect ref_dBm per burst once
for burst_sir in by_sir[0]:    # iterate over every SIR bucket
    ref_store.extend( len(by_sir[0][burst_sir]) * [burst_sir] )  # dummies

ref_all = np.asarray(ref_store, dtype=float)

for m in range(len(nets)):
    err_all = np.hstack([by_sir[m][s] for s in by_sir[m]])
    pct_means.append(mean_pct_err(err_all, ref_all))

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


