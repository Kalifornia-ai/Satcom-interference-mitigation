#!/usr/bin/env python3
"""
evaluate_beacon_net_multi.py  –  compare ≥ 1 checkpoints (CNN / Hybrid / LSTM /
TorchScript).  Every metric is now reported in the **raw (un-normalised) scale**.
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
from lstm    import LSTMSeperatorSingle, LSTMSingleSource      # two lstm flavours
# ───────────────– CLI ──────────────────────────────────────────────────
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--data",   required=True, type=Path)
ap.add_argument("--ckpts",  nargs="+",    required=True, type=Path)
ap.add_argument("--outdir", default="eval_multi", type=Path)
ap.add_argument("--meta",   default="cw_power_train_meta.json", type=Path)
ap.add_argument("--len-override", type=int, default=None)
args = ap.parse_args(); args.outdir.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CMAP   = plt.get_cmap("tab10")
dBm_fac= 10.0 / math.log(10)
GAIN   = 0.375                    # |Hann|² loss for FFT 3-bin trick
to_dB  = lambda x: 20*np.log10(x)

# ───────── util helpers ────────────────────────────────────────────────
def gain_to_dBm(g: complex) -> float:      # raw-scale carrier → dBm
    return dBm_fac * math.log((abs(g)**2)/1e-3)

def fft_3bin_amp(x: np.ndarray) -> float:  # complex burst → linear volts
    X = fft(x*np.hanning(len(x))) / len(x)
    return np.sqrt(np.sum(np.abs(X[[0,1,-1]])**2) / GAIN)

def seq_to_gain(y: torch.Tensor) -> complex:   # (B,T,2) → scalar complex
    return torch.complex(y[...,0].mean(1), y[...,1].mean(1))

def freq_shift_np(x: np.ndarray, f_shift: float, fs: float = 10e6):
    n = np.arange(len(x), dtype=np.float64)
    return x * np.exp(2j*np.pi*f_shift*n/fs).astype(np.complex64)

# ───────── recognise state-dicts ───────────────────────────────────────
def identify(sd):
    k=list(sd)
    if "front.0.weight"              in k: return BeaconPowerCNN
    if "res.0.c1.weight"             in k: return HybridBeaconEstimator
    if "lstm.0.weight_ih_l0"         in k: return LSTMSeperatorSingle
    if "lstm_layers.0.weight_ih_l0"  in k: return LSTMSingleSource
    raise RuntimeError("unrecognised state-dict")

# ───────── load checkpoints ────────────────────────────────────────────
nets, labels, colors, in_len, time_major, needs_shift = [],[],[],[],[],[]
for i, ck in enumerate(args.ckpts):
    try:                       # state-dict
        sd  = torch.load(ck, map_location=DEVICE, weights_only=True)
        cls = identify(sd); net = cls().to(DEVICE); net.load_state_dict(sd)
        time_major.append(cls in (LSTMSeperatorSingle,LSTMSingleSource))
    except Exception:          # TorchScript
        net = torch.jit.load(ck, map_location=DEVICE); time_major.append(False)
    nets  .append(net.eval())
    labels.append(ck.stem)
    colors.append(CMAP(i%10))
    needs_shift.append("sine" in ck.stem.lower())     # special 200 kHz shift
    in_len.append(args.len_override or 1000)

# ───────── build test-set file list ────────────────────────────────────
meta = json.loads((args.data/args.meta).read_text())
test_idx = np.array(meta["test_idx"], dtype=int)
files = np.array(sorted(args.data.rglob("*.npz")))[test_idx]
print(f"Evaluating {len(files)} bursts …")

# ───────── containers ─────────────────────────────────────────────────
by_sir        = [defaultdict(list) for _ in nets]
by_sir_fft    = defaultdict(list)
by_cw_qp      = [defaultdict(lambda: defaultdict(list)) for _ in nets]
by_cw_qp_fft  = defaultdict(lambda: defaultdict(list))
plotted_sir   = set()

re_cw  = re.compile(r"Sine_50002[-_]?(-?\d+)dBm")
re_qps = re.compile(r"QPSK_5G[-_]?(-?\d+)dBm")
def levels(folder):
    cw  = int(re_cw .search(folder).group(1))
    qps = int(re_qps.search(folder).group(1))
    return cw, qps, -qps-cw

def db_spectrum(sig):
    N=len(sig); S=np.fft.fftshift(np.fft.fft(sig*np.hanning(N)))/N
    f=np.fft.fftshift(np.fft.fftfreq(N,d=1/10e6))/1e6
    return f, 20*np.log10(np.abs(S)+1e-18)

# ───────── iterate over test bursts ────────────────────────────────────
for npz in tqdm(files, unit="file"):
    cw_dBm,qpsk_dBm,sir = levels(npz.parent.name)
    with np.load(npz,allow_pickle=True) as z:
        x_raw = z["x"].astype(np.complex64)
        g_ref = z["meta"].item()["pristine_gain"]       # ***raw scale***
    ref_dBm = gain_to_dBm(g_ref)

    rms = np.sqrt(np.mean(np.abs(x_raw)**2))            # keep for later
    x_norm = x_raw / rms                                # what nets expect

    preds=[]                                            # raw-scale dBm list
    with torch.no_grad():
        for m,net in enumerate(nets):
            N=in_len[m]
            burst = x_norm
            if N<len(burst):  s=(len(burst)-N)//2; burst=burst[s:s+N]
            elif N>len(burst):p=(N-len(burst))//2;  burst=np.pad(burst,(p,N-len(burst)-p))

            if needs_shift[m]:                          # 200 kHz for “sine” CNN
                burst = freq_shift_np(burst, 2.0e5) / rms   # still norm. magnitude

            xt = ( torch.tensor(np.stack([burst.real,burst.imag],-1 if time_major[m] else 0),
                                dtype=torch.float32)
                   .unsqueeze(0).to(DEVICE) )
            out=net(xt);  out = out[0] if isinstance(out,tuple) else out
            if isinstance(out,dict): out = out["gain"]

            # -------- convert prediction → raw dBm -------------------
            if out.ndim==3:      # (B,T,2)
                g_hat = seq_to_gain(out)[0].cpu().item()*rms
                pred  = gain_to_dBm(g_hat)
            elif out.ndim==2 and out.shape[-1]==2:   # (B,2)
                g_hat = complex(out[0,0].item(), out[0,1].item())*rms
                pred  = gain_to_dBm(g_hat)
            else:                # scalar dBm (normalised)  → add 20 log10(rms)
                pred  = out.cpu().item() + to_dB(rms)
            preds.append(pred)

    # -------- baseline FFT (done on raw burst) -------------------------
    fft_err = gain_to_dBm(fft_3bin_amp(x_raw)*np.exp(1j*np.angle(g_ref))) - ref_dBm
    by_sir_fft[sir].append(fft_err)
    by_cw_qp_fft[cw_dBm][qpsk_dBm].append(fft_err)

    # -------- per-network error buckets --------------------------------
    for m,p in enumerate(preds):
        err = p - ref_dBm
        by_sir   [m][sir]           .append(err)
        by_cw_qp [m][cw_dBm][qpsk_dBm].append(err)

    # -------- PSD figure (once per SIR) -------------------------------
    if sir not in plotted_sir:
        f_MHz,P_mix=db_spectrum(x_raw)
        y_gt = np.full_like(x_raw, g_ref);     _,P_gt  = db_spectrum(y_gt)
        y_hat= np.full_like(x_raw,
                 math.sqrt(1e-3*10**(preds[0]/10))*np.exp(1j*np.angle(g_ref)))
        _,P_hat= db_spectrum(y_hat)
        y_fft = np.full_like(x_raw,
                 fft_3bin_amp(x_raw)*np.exp(1j*np.angle(g_ref)))
        _,P_fft= db_spectrum(y_fft)
        for rng,tag in [((-5,5),'full'),((-0.1,0.1),'zoom')]:
            fig,ax=plt.subplots(figsize=(6,3))
            ax.plot(f_MHz,P_mix,lw=.7,label='mixture')
            ax.plot(f_MHz,P_gt,         label='ground truth')
            ax.plot(f_MHz,P_hat,'--',   label=f'estimate ({labels[0]})')
            ax.plot(f_MHz,P_fft,':',    label='FFT 3-bin')
            ax.set(xlabel='Freq (MHz)',ylabel='dB re 1 mW',xlim=rng,ylim=(-150,0),
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


