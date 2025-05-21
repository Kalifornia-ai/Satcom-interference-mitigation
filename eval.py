#!/usr/bin/env python3
"""
evaluate_beacon_net.py – full pipeline (all plots + CSV)

Now a single-pass version (no duplicated metrics, no extra loops).
"""

import argparse, csv, math, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.amp import autocast
from scipy.fft import fft
from scipy.stats import wilcoxon

from model import BeaconPowerCNN         # ← adjust import if needed
# ────────────────────────── configuration ────────────────────────────────
FOLDERS = [
    "Sine_50002-50dBm_QPSK_5G_-10dBm", "Sine_50002-45dBm_QPSK_5G_-10dBm",
    "Sine_50002-40dBm_QPSK_5G_-10dBm", "Sine_50002-45dBm_QPSK_5G_-20dBm",
    "Sine_50002-40dBm_QPSK_5G_-20dBm", "Sine_50002-25dBm_QPSK_5G_-10dBm",
    "Sine_50002-20dBm_QPSK_5G_-10dBm", "Sine_50002-25dBm_QPSK_5G_-20dBm",
    "Sine_50002-20dBm_QPSK_5G_-20dBm", "Sine_50002-25dBm_QPSK_5G_-30dBm",
    "Sine_50002-20dBm_QPSK_5G_-30dBm", "Sine_50002-25dBm_QPSK_5G_-40dBm",
    "Sine_50002-10dBm_QPSK_5G_-30dBm", "Sine_50002-25dBm_QPSK_5G_-50dBm",
    "Sine_50002-10dBm_QPSK_5G_-40dBm", "Sine_50002-10dBm_QPSK_5G_-50dBm",
]
PAT_CW   = re.compile(r"Sine_50002[-_]?(-?\d+)dBm")
PAT_QPSK = re.compile(r"QPSK_5G[-_]?(-?\d+)dBm")

# ────────────────────────── CLI & paths ──────────────────────────────────
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--data",   required=True,  type=Path, help="dataset root")
ap.add_argument("--ckpt",   required=True,  type=Path, help="trained .pt file")
ap.add_argument("--outdir", default="eval_figs", type=Path, help="output folder")
ap.add_argument("--fs", type=float, default=10e6, help="sample-rate [Hz]")
args = ap.parse_args(); args.outdir.mkdir(parents=True, exist_ok=True)

# ────────────────────────── model ────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
net = BeaconPowerCNN().to(DEVICE)
net.load_state_dict(torch.load(args.ckpt, map_location=DEVICE, weights_only=True))
net.eval()

dBm_factor = 10.0 / math.log(10)
def gain_to_dBm(g):
    return dBm_factor * math.log((abs(g)**2) / 1e-3)

# ────────────────────────── helpers ──────────────────────────────────────
GAIN       = 0.375                     # 3-bin Hann correction
F_SHIFT_HZ = 200_000                   # shift for display

def fft_3bin_amp(x: np.ndarray) -> float:
    N  = len(x)
    X  = fft(x * np.hanning(N)) / N
    return np.sqrt(np.sum(np.abs(X[[0, 1, -1]])**2) / GAIN)

def db_spectrum(sig: np.ndarray, fs: float, ref_rms: float):
    N   = len(sig)
    S   = np.fft.fftshift(np.fft.fft(sig * np.hanning(N))) / N
    mag = 20*np.log10(np.abs(S)/(ref_rms+1e-18))
    f   = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))/1e6
    return f, mag

amp_err_db = lambda p,t: 20*math.log10(abs(p)/(abs(t)+1e-12))
ph_err_deg = lambda p,t: np.rad2deg(np.angle(p)-np.angle(t))

def parse_levels(folder):
    cw   = int(PAT_CW.search(folder).group(1))
    qpsk = int(PAT_QPSK.search(folder).group(1))
    return cw, qpsk, -qpsk - cw

# ────────────────────────── main containers ─────────────────────────────
by_sir     = defaultdict(lambda: defaultdict(list))
by_sir_fft = defaultdict(list)
plotted_sir = set()                     # track which SIRs already have PNGs

# ────────────────────────── single pass over files ───────────────────────
files = [p for name in FOLDERS for p in (args.data/name).glob("*.npz")]
for f in tqdm(files, unit="file"):
    cw_dbm, _q_dbm, sir = parse_levels(f.parent.name)

    with np.load(f, allow_pickle=True) as z:
        x   = z["x"]                                   # raw burst (1000,)
        g_p = z["meta"].item()["pristine_gain"]        # complex64
    ref_dBm = gain_to_dBm(g_p)

  

    # ---------- network forward ----------------------------------------
    x_t = torch.view_as_real(torch.from_numpy(x)).unsqueeze(0).to(DEVICE)
    with torch.no_grad(), autocast(device_type=DEVICE):
        rms  = np.sqrt(np.mean(np.abs(x)**2))
        x_n  = x / rms                                     # unit RMS
        x_t  = torch.tensor(np.stack([x_n.real, x_n.imag], 0))  # (2,1000)
    
    pred_dBm = net(x_t.unsqueeze(0).to(DEVICE)).item()
    g_hat = complex(g_hat_t.real.item(), g_hat_t.imag.item())

    # ---------- baseline & metrics -------------------------------------
    amp_fft = fft_3bin_amp(x)
    by_sir[sir]['amp'].append(pred_dBm - ref_dBm)
    
    #by_sir[sir]['df'] .append(df_hat - df_gt)
    by_sir[sir]['cw'] .append(cw_dbm)
    by_sir_fft[sir].append(20*math.log10(amp_fft/(abs(g_ref)+1e-12)))
    # keep raw refs for later drifting/burst tests
    by_sir[sir]['x_raw'].append(x)
    by_sir[sir]['g_ref'].append(g_ref)

    # ---------- one-PNG-per-SIR ----------------------------------------
    if sir not in plotted_sir:
        t   = np.arange(len(x)) / args.fs
        rot = np.exp(2j*np.pi*F_SHIFT_HZ*t)
        x_sh, est_sh, gt_sh = x*rot, g_hat*rot, g_ref*rot
        rms_mix = np.sqrt(np.mean(np.abs(x_sh)**2))

        N_td = min(2048, len(x_sh))
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(np.real(x_sh[:N_td])/rms_mix,  label="mixture", lw=.8)
        ax.plot(np.real(est_sh[:N_td])/rms_mix, label="estimate")
        ax.plot(np.real(gt_sh[:N_td])/rms_mix, label="ground truth", ls="--")
        ax.set(title=f"Time (+200 kHz) – SIR {sir} dB",
               xlabel="sample", ylabel="norm. amp"); ax.grid(ls=":")
        ax.legend(frameon=False, ncol=3)
        fig.tight_layout(); fig.savefig(args.outdir/f"td_SIR{sir}.png", dpi=220); plt.close(fig)

        f_MHz, spec_mix = db_spectrum(x_sh, args.fs, rms_mix)
        _,     spec_est = db_spectrum(est_sh, args.fs, rms_mix)
        _,     spec_gt  = db_spectrum(gt_sh,  args.fs, rms_mix)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(f_MHz, spec_mix, label="mixture", lw=.8)
        ax.plot(f_MHz, spec_est, label="estimate")
        ax.plot(f_MHz, spec_gt,  label="ground truth", ls="--")
        ax.set(title=f"Spectrum (+200 kHz) – SIR {sir} dB",
               xlabel="Frequency (MHz)", ylabel="dB re RMS",
               xlim=(-args.fs/2e6, args.fs/2e6)); ax.grid(ls=":")
        ax.set_ylim(-150, 0) 
        ax.legend(frameon=False, ncol=3)
        fig.tight_layout(); fig.savefig(args.outdir/f"spec_SIR{sir}.png", dpi=220); plt.close(fig)

        plotted_sir.add(sir)

# ───────────────── colour-map range & sorted SIR list ────────────────────
sirs  = np.array(sorted(by_sir))
vmin  = min(lvl for s in by_sir for lvl in by_sir[s]['cw'])
vmax  = max(lvl for s in by_sir for lvl in by_sir[s]['cw'])
shift = 0.35

# ───────── colour-map range & stats derived from by_sir ─────────
sirs  = np.array(sorted(by_sir))
vmin  = min(lvl for s in by_sir for lvl in by_sir[s]['cw'])
vmax  = max(lvl for s in by_sir for lvl in by_sir[s]['cw'])
shift = 0.35

# --- MISSING LINES: add mean curves for the two scatter plots ---
ml_means  = [np.mean(by_sir[s]['amp'])   for s in sirs]
fft_means = [np.mean(by_sir_fft[s])      for s in sirs]

# ────────────── 1) ML scatter --------------------------------------------
fig, ax = plt.subplots(figsize=(6.5,4))
for sir in sirs:
    cw_levels = np.array(by_sir[sir]['cw'])
    # colour is mapped via "c=" so matplotlib handles the normalization
    sc = ax.scatter(
        np.full_like(cw_levels, sir-shift),
        by_sir[sir]['amp'],
        c=cw_levels, cmap='turbo', vmin=vmin, vmax=vmax,
        edgecolors='k', linewidths=0.3,
        marker='o', s=60, alpha=0.8)

# mean curve on top
ax.plot(sirs-shift, ml_means, color='C0', lw=2.0, label='mean')

ax.set(title='ML Beacon Net', xlabel='SIR (dB)', ylabel='ΔA (dB)')
ax.grid(ls=':'); ax.axhline(0, color='grey', lw=.8)
fig.colorbar(sc, ax=ax, label='CW power (dBm)')
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(args.outdir/'amp_vs_SIR_ML.png', dpi=220)

# ────────────── 2) FFT scatter -------------------------------------------
fig, ax = plt.subplots(figsize=(6.5,4))
for sir in sirs:
    cw_levels = np.array(by_sir[sir]['cw'])
    # colour by CW power, same scale as ML plot
    sc_fft = ax.scatter(
        np.full_like(cw_levels, sir+shift),
        by_sir_fft[sir],
        c=cw_levels, cmap='turbo', vmin=vmin, vmax=vmax,
        marker='^', s=75, alpha=.8, edgecolors='k', linewidths=0.3)

# mean trend
ax.plot(sirs+shift, fft_means, color='k', lw=2.0, ls='--', label='mean')
ax.set(title='FFT 3‑bin Baseline', xlabel='SIR (dB)', ylabel='ΔA (dB)')
ax.grid(ls=':'); ax.axhline(0, color='grey', lw=.8)
fig.colorbar(sc_fft, ax=ax, label='CW power (dBm)')
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(args.outdir/'amp_vs_SIR_FFT.png', dpi=220)

# ────────────── 1a) ML scatter – zoom (SIR ≥ –10 dB) --------------------
sirs_hi = sirs[sirs >= -10]
fig, ax = plt.subplots(figsize=(6.5,4))
for sir in sirs_hi:
    cw_levels = np.array(by_sir[sir]['cw'])
    sc_ml_hi = ax.scatter(
        np.full_like(cw_levels, sir),
        by_sir[sir]['amp'],
        c=cw_levels, cmap='turbo', vmin=vmin, vmax=vmax,
        edgecolors='k', linewidths=0.3,
        marker='o', s=60, alpha=0.8)
ax.plot(sirs_hi, [ml_means[list(sirs).index(s)] for s in sirs_hi],
        color='C0', lw=2.0, label='mean')
ax.set_xlabel('SIR (dB)'); ax.set_ylabel('ΔA (dB)')
ax.set_title('ML ΔA – zoomed  (SIR ≥ –10 dB)')
ax.set_xlim(-10.5, sirs_hi.max()+2)
ax.set_ylim(-3, 3)
ax.grid(ls=':'); ax.axhline(0, color='grey', lw=.8)
fig.colorbar(sc_ml_hi, ax=ax, label='CW power (dBm)')
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(args.outdir/'amp_vs_SIR_ML_hiSIR.png', dpi=220)

# ────────────── 2a) FFT scatter – zoom (SIR ≥ –10 dB) -------------------
fig, ax = plt.subplots(figsize=(6.5,4))
for sir in sirs_hi:
    cw_levels = np.array(by_sir[sir]['cw'])
    sc_fft_hi = ax.scatter(
        np.full_like(cw_levels, sir),
        by_sir_fft[sir],
        c=cw_levels, cmap='turbo', vmin=vmin, vmax=vmax,
        marker='^', s=75, alpha=.8, edgecolors='k', linewidths=0.3)
ax.plot(sirs_hi, [fft_means[list(sirs).index(s)] for s in sirs_hi],
        color='k', lw=2.0, ls='--', label='mean')
ax.set_xlabel('SIR (dB)'); ax.set_ylabel('ΔA (dB)')
ax.set_title('FFT 3‑bin ΔA – zoomed  (SIR ≥ –10 dB)')
ax.set_xlim(-10.5, sirs_hi.max()+2)
ax.set_ylim(-3, 3)
ax.grid(ls=':'); ax.axhline(0, color='grey', lw=.8)
fig.colorbar(sc_fft_hi, ax=ax, label='CW power (dBm)')
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(args.outdir/'amp_vs_SIR_FFT_hiSIR.png', dpi=220)

# ────────────── 3) ML vs FFT box‑plot ------------------------------------
fig, ax = plt.subplots(figsize=(7,4))
shift_box, width = 1.2, 0.9
ml_data  = [by_sir[s]['amp'] for s in sirs]
fft_data = [by_sir_fft[s]    for s in sirs]

ax.boxplot(ml_data,  positions=sirs-shift_box, widths=width, showfliers=False,
           patch_artist=True, boxprops=dict(facecolor='C0', alpha=.6),
           medianprops=dict(color='k'))
ax.boxplot(fft_data, positions=sirs+shift_box, widths=width, showfliers=False,
           patch_artist=True,
           boxprops=dict(facecolor='white', edgecolor='k', hatch='///'),
           medianprops=dict(color='k'))
ax.set_xticks(sirs); ax.set_xticklabels([f"{int(s)}" for s in sirs])
ax.set_xlabel('SIR (dB)'); ax.set_ylabel('ΔA (dB)'); ax.grid(ls=':')
from matplotlib.patches import Patch
handles = [
    Patch(facecolor='C0', edgecolor='C0', label='ML'),
    Patch(facecolor='white', edgecolor='k', hatch='///', label='FFT 3-bin')
]
ax.legend(handles=handles, frameon=False, loc='upper right')
fig.tight_layout(); fig.savefig(args.outdir/'amp_box.png', dpi=220)

# ────────────── 4) FFT – ML difference panel -----------------------------
fig, ax = plt.subplots(figsize=(7,2.8))
diff_data = [np.array(f) - np.array(m) for f, m in zip(fft_data, ml_data)]
ax.boxplot(diff_data, positions=sirs, widths=0.9, showfliers=False, whis=[5,95],
           patch_artist=True, boxprops=dict(facecolor='lightgrey', alpha=.65),
           medianprops=dict(color='k'))
ax.axhspan(-1, 1, color='lightgreen', alpha=.15)
ax.text(sirs[1], 1.4, 'spec ±1 dB', fontsize=9)
ax.axhline(0, color='k', lw=.8)
ax.set(title='Negative ⇒ FFT better   |   Positive ⇒ ML better',
       xlabel='SIR (dB)', ylabel='FFT – ML  ΔA (dB)')
ax.set_ylim(-10, 5)
ax.grid(ls=':')
fig.tight_layout(); fig.savefig(args.outdir/'amp_diff_box.png', dpi=220)

# ────────────── 4a) *Hi‑SIR* box‑plot (SIR ≥ –10 dB) ----------------------
sirs_hi   = sirs[sirs >= -10]
ml_hi     = [by_sir[s]['amp'] for s in sirs_hi]
fft_hi    = [by_sir_fft[s]    for s in sirs_hi]
fig, ax = plt.subplots(figsize=(6.5,3.5))
ax.boxplot(ml_hi,  positions=sirs_hi-0.4, widths=0.7, showfliers=False,
           patch_artist=True, boxprops=dict(facecolor='C0', alpha=.6),
           medianprops=dict(color='k'))
ax.boxplot(fft_hi, positions=sirs_hi+0.4, widths=0.7, showfliers=False,
           patch_artist=True, boxprops=dict(facecolor='white', edgecolor='k', hatch='///'),
           medianprops=dict(color='k'))
ax.set_xticks(sirs_hi); ax.set_xlabel('SIR (dB)')
ax.set_ylabel('ΔA (dB)'); ax.grid(ls=':')
ax.set_title('Amplitude error – zoomed view  (SIR ≥ –10 dB)')
ax.set_ylim(-2, 3)
ax.legend(handles=handles, frameon=False, loc='upper right')
fig.tight_layout(); fig.savefig(args.outdir/'amp_box_hiSIR.png', dpi=220)


# ────────────── 5) Phase scatter -----------------------------------------
fig, ax = plt.subplots(figsize=(6,4))
for sir in sirs:
    pts = ax.scatter(
        np.full_like(by_sir[sir]['ph'], sir),
        by_sir[sir]['ph'],
        c=by_sir[sir]['cw'], cmap='turbo', vmin=vmin, vmax=vmax,
        s=10, alpha=.4)

means_ph = [np.mean(by_sir[s]['ph']) for s in sirs]
ax.plot(sirs, means_ph, 'k-o', lw=1.2, ms=4, label='mean')
ax.set_xlabel('SIR (dB)'); ax.set_ylabel('Δφ (deg)')
ax.grid(ls=':'); ax.axhline(0, color='grey', lw=.8)
sm2 = plt.cm.ScalarMappable(cmap='turbo', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm2.set_array([])
fig.colorbar(sm2, ax=ax, label='CW power (dBm)')
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(args.outdir/'phase_vs_SIR.png', dpi=220)

# ────────────── 6) Δf ribbon ---------------------------------------------
# means_df = [np.mean(by_sir[s]['df']) for s in sirs]
# stds_df  = [np.std (by_sir[s]['df']) for s in sirs]
# fig, ax = plt.subplots(figsize=(6,4))
# ax.plot(sirs, means_df, 'k-o', label='mean Δf')
# ax.fill_between(sirs, np.array(means_df)-stds_df, np.array(means_df)+stds_df, alpha=.2)
# ax.set_xlabel('SIR (dB)'); ax.set_ylabel('Δf (Hz)')
# ax.grid(ls=':'); ax.legend(frameon=False)
# fig.tight_layout(); fig.savefig(args.outdir/'df_vs_SIR.png', dpi=220)

# ────────────── 7) NEW: ΔA vs CW‑power scatter ---------------------------
fig, ax = plt.subplots(figsize=(6.5,4))
all_cw   = []
all_da   = []
all_sir  = []
for sir in sirs:
    all_cw.extend(by_sir[sir]['cw'])
    all_da.extend(by_sir[sir]['amp'])   # ML errors
    all_sir.extend([sir]*len(by_sir[sir]['amp']))

sc2 = ax.scatter(all_cw, all_da,
                 c=all_sir, cmap='viridis', vmin=sirs.min(), vmax=sirs.max(),
                 s=35, alpha=.7, edgecolors='none')
ax.set_xlabel('CW power (dBm)')
ax.set_ylabel('ΔA  ML estimator (dB)')
ax.set_title('ΔA vs CW level – colour = SIR (dB)')
ax.grid(ls=':'); ax.axhline(0, color='grey', lw=.8)
fig.colorbar(sc2, ax=ax, label='SIR (dB)')
fig.tight_layout(); fig.savefig(args.outdir/'dA_vs_CW.png', dpi=220)



# ────────────── 9) Frequency‑drift sweep (vectorised) --------------------
# Use a 10 % random subset to keep runtime low
SAMPLE_FRAC = 0.10
all_pairs   = [(s, i) for s in sirs for i in range(len(by_sir[s]['x_raw']))]
np.random.shuffle(all_pairs)
keep_pairs  = all_pairs[:int(len(all_pairs)*SAMPLE_FRAC)]

# Pack the subset into two big tensors once (CPU → GPU one time)
# build a proper 3‑D batch tensor: (B, N, 2)
xs = torch.stack([
    torch.view_as_real(torch.from_numpy(by_sir[s]['x_raw'][i]))
    for s, i in keep_pairs
]).to(DEVICE)
grefs= torch.tensor([abs(by_sir[s]['g_ref'][i]) for s,i in keep_pairs], device=DEVICE)
Nrec = xs.shape[-2]

DRIFTS   = torch.linspace(-15, 15, 13, device=DEVICE)   # bins
rot_idx  = torch.arange(Nrec, device=DEVICE)
ml_drift, fft_drift = [], []

with torch.no_grad(), autocast(device_type=DEVICE):
    for d in DRIFTS:
        rot = torch.exp(2j*math.pi*d*rot_idx/Nrec)
        x_rot = torch.view_as_complex(xs) * rot          # broadcast over batch
        # ---------- ML path ----------
        out = net(torch.view_as_real(x_rot))
        g_hat = torch.view_as_complex(out['gain']).flatten()
        
        ml_drift.append(torch.mean(20*torch.log10(torch.abs(g_hat)/grefs)).item())
                # ---------- FFT path ----------
        amps_fft = []
        for x_cpu, g_ref_scalar in zip(x_rot.cpu().numpy(), grefs.cpu().numpy()):
            amps_fft.append(amp_err_db(fft_3bin_amp(x_cpu), g_ref_scalar))
        fft_drift.append(float(np.mean(amps_fft)))

# convert drift axis to numpy for plotting
DRIFTS_np = DRIFTS.cpu().numpy()
fig, ax = plt.subplots(figsize=(5,4))
ax.plot(DRIFTS_np, ml_drift,  'o-', label='ML')
ax.plot(DRIFTS_np, fft_drift, '^--', label='FFT 3‑bin')
ax.set_xlabel('Carrier offset (bins)'); ax.set_ylabel('Mean ΔA (dB)')
ax.set_title('Robustness to frequency drift (10 % sample)')
ax.grid(ls=':'); ax.legend()
fig.tight_layout(); fig.savefig(args.outdir/'drift_sweep.png', dpi=220)

# ────────────── 10) Burst‑noise robustness (sampled) --------------------
np.random.seed(0)
SAMPLE_FRAC_BURST = 0.10
ml_burst, fft_burst = [], []
for s in sirs:
    idx_subset = np.random.choice(len(by_sir[s]['x_raw']),
                                  max(1, int(len(by_sir[s]['x_raw'])*SAMPLE_FRAC_BURST)),
                                  replace=False)
    errs_ml, errs_fft = [], []
    for idx in idx_subset:
        x_raw, g_ref = by_sir[s]['x_raw'][idx], by_sir[s]['g_ref'][idx]
        x_b = x_raw.copy()
        burst_idx = np.random.choice(len(x_b), int(0.001*len(x_b)), replace=False)
        x_b[burst_idx] *= 10
        # ML
        x_t = torch.view_as_real(torch.from_numpy(x_b)).unsqueeze(0).to(DEVICE)
        with torch.no_grad(), autocast(device_type=DEVICE):
            g_hat = torch.view_as_complex(net(x_t.float())['gain'].cpu().view(1,1,2))[0,0]
        errs_ml.append(amp_err_db(g_hat, g_ref))
        # FFT
        errs_fft.append(amp_err_db(fft_3bin_amp(x_b), g_ref))
    ml_burst.append(np.sqrt(np.mean(np.square(errs_ml))))
    fft_burst.append(np.sqrt(np.mean(np.square(errs_fft))))

fig, ax = plt.subplots(figsize=(5,4))
ax.plot(sirs, ml_burst, 'o-', label='ML RMSE')
ax.plot(sirs, fft_burst,'^--', label='FFT 3‑bin RMSE')
ax.set_xlabel('SIR (dB)'); ax.set_ylabel('ΔA RMSE (dB)')
ax.set_title('Burst noise (0.1 % ×10) – 10 % sample')
ax.grid(ls=':'); ax.legend()
fig.tight_layout(); fig.savefig(args.outdir/'burst_noise.png', dpi=220)




# ────────────────────────────────────────────────────────────────────────

# ────────────── 11) summary.csv + Wilcoxon p-values ----------------------
ml_data_list  = [by_sir[s]['amp'] for s in sirs]
fft_data_list = [by_sir_fft[s]    for s in sirs]

pvals = [wilcoxon(f, m).pvalue for f, m in zip(fft_data_list, ml_data_list)]
csv_path = args.outdir/'summary.csv'
with open(csv_path, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['SIR','µ_ML','σ_ML','µ_FFT','σ_FFT','RMSE_ML','RMSE_FFT','p_Wilcoxon'])
    for s, ml_arr, fft_arr, p in zip(sirs, ml_data_list, fft_data_list, pvals):
        ml_arr, fft_arr = np.asarray(ml_arr), np.asarray(fft_arr)
        w.writerow([
            int(s),
            f'{ml_arr.mean():.3f}',  f'{ml_arr.std():.3f}',
            f'{fft_arr.mean():.3f}', f'{fft_arr.std():.3f}',
            f'{np.sqrt((ml_arr**2).mean()):.3f}',
            f'{np.sqrt((fft_arr**2).mean()):.3f}',
            f'{p:.2e}'
        ])

# ────────────── console summary -----------------------------------------
print(f'✓ Figures & {csv_path.name} written to {args.outdir.resolve()}')
print('Paired Wilcoxon p‑values by SIR:')
print({int(s): f'{p:.2e}' for s, p in zip(sirs, pvals)})