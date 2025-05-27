#!/usr/bin/env python3
"""
plot_iq_waveforms.py – one PNG per unique (CW, QPSK) power pair.

Outputs, for every (CW, QPSK) level present in traced_pairs.csv:

    CW{±dB}_QP{±dB}_wave.png   # mixture, clean CW, estimate – time domain
    CW{±dB}_QP{±dB}_psd.png    # Welch PSD (dBm/Hz, ±1.5 MHz)
    CW{±dB}_QP{±dB}_const.png  # constellation scatter of I/Q
"""
from __future__ import annotations
import argparse, csv, re, sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from scipy.signal import welch, get_window
from numpy.fft import fftshift

# ─────────── Matplotlib (IEEE) defaults ───────────────────────────────
mpl.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 0.6,
    }
)

# ─────────── optional model classes (ignore if missing) ───────────────
try:
    from model import BeaconPowerCNN          # type: ignore
    from resmodel import HybridBeaconEstimator  # type: ignore
    from lstm import LSTMSingleSource, LSTMSeperatorSingle  # type: ignore
except Exception:
    BeaconPowerCNN = HybridBeaconEstimator = LSTMSingleSource = LSTMSeperatorSingle = None  # type: ignore

# ─────────── small helpers ────────────────────────────────────────────
def read_csv_iq(p: Path) -> np.ndarray:
    a = np.loadtxt(p, delimiter=",", skiprows=1, dtype=np.float32)
    return (a[:, 0] + 1j * a[:, 1]).astype(np.complex64)

def freq_shift(x: np.ndarray, fs: float, f0: float) -> np.ndarray:
    if f0 == 0:
        return x
    n = np.arange(len(x), dtype=np.float32)
    return x * np.exp(+1j * 2 * np.pi * f0 * n / fs)

def welch_psd_dbm(x: np.ndarray, fs: float,
                  nperseg: int = 1000, window: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    f, Pxx = welch(x, fs=fs, window=get_window(window, nperseg),
                   nperseg=nperseg, noverlap=nperseg//2,
                   scaling="density", return_onesided=False)
    f = fftshift(f) / 1e3                      # kHz
    Pxx = fftshift(Pxx)
    return f, 10*np.log10(Pxx/1e-3 + 1e-18)

def identify_state_dict(sd: dict) -> type[torch.nn.Module]:
    if BeaconPowerCNN and "front.0.weight"           in sd: return BeaconPowerCNN
    if HybridBeaconEstimator and "res.0.c1.weight"   in sd: return HybridBeaconEstimator
    if LSTMSeperatorSingle and "lstm.0.weight_ih_l0" in sd: return LSTMSeperatorSingle
    if LSTMSingleSource   and "lstm_layers.0.weight_ih_l0" in sd: return LSTMSingleSource
    raise RuntimeError("Unknown checkpoint format")

# ─────────── main script ──────────────────────────────────────────────
def main() -> None:
    cli = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cli.add_argument("dataset_root", type=Path)
    cli.add_argument("measured_root", type=Path)
    cli.add_argument("out_dir",      type=Path)
    cli.add_argument("--pairs_csv", required=True, type=Path)
    grp = cli.add_mutually_exclusive_group(required=True)
    grp.add_argument("--ckpt",      type=Path)
    grp.add_argument("--pred_root", type=Path)
    cli.add_argument("--fs",    default=10e6, type=float)
    cli.add_argument("--first", default=2000, type=int)
    cli.add_argument("--len",   default=1000, type=int)
    cli.add_argument("--dpi",   default=600,  type=int)
    cli.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = cli.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── load checkpoint (optional) ─────────────────────────────────────
    net, time_major = None, False
    if args.ckpt:
        try:
            sd  = torch.load(args.ckpt, map_location=args.device, weights_only=True)  # type: ignore[arg-type]
            Net = identify_state_dict(sd)
            net = Net().to(args.device)                                              # type: ignore[operator]
            net.load_state_dict(sd)
            time_major = Net in (LSTMSeperatorSingle, LSTMSingleSource)
        except Exception:
            net = torch.jit.load(args.ckpt, map_location=args.device)
        net.eval()
    elif not args.pred_root.exists():
        sys.exit("--pred_root not found and no --ckpt given")

    # ── read traced_pairs.csv ──────────────────────────────────────────
    pairs: List[Tuple[str,str]] = []
    with args.pairs_csv.open() as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            mix = row.get("mixture"); cln = row.get("clean_file")
            if mix and cln: pairs.append((mix.strip(), cln.strip()))
    if not pairs: sys.exit("pairs_csv empty or malformed")

    # ── helpers --------------------------------------------------------
    clean_lookup: Dict[str,Path] = {p.name: p for p in
        args.measured_root.glob("Sine_50002*[-_]*dBm*/**/*.csv") if "QPSK" not in p.parts[-2].upper()}

    pat_cw  = re.compile(r"Sine_50002[-_]?([+-]?\d+)dBm", re.I)
    pat_qps = re.compile(r"QPSK_5G[-_]?([+-]?\d+)dBm",   re.I)
    colours = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

    done: Set[Tuple[int,int]] = set(); plotted = 0

    for mix_rel, clean_csv in pairs:
        mix_npz = args.dataset_root / mix_rel
        if not mix_npz.exists(): continue
        m_cw, m_qp = pat_cw.search(mix_npz.parent.name), pat_qps.search(mix_npz.parent.name)
        if not (m_cw and m_qp): continue
        cw, qp = int(m_cw.group(1)), int(m_qp.group(1))
        if (cw,qp) in done: continue
        done.add((cw,qp))

        # mixture
        mix_raw = np.load(mix_npz, allow_pickle=True)["x"].astype(np.complex64)
        rms     = np.sqrt(np.mean(np.abs(mix_raw)**2))
        n_view  = len(mix_raw) if args.first==0 else min(args.first,len(mix_raw))
        t_ax    = np.arange(n_view)

        # clean CW
        csv_path = clean_lookup.get(clean_csv)
        if not csv_path: 
            print("[skip] clean CSV missing:", clean_csv, file=sys.stderr); continue
        clean = read_csv_iq(csv_path)[:len(mix_raw)]

        # estimate
        if net:
            burst = mix_raw / rms
            N = args.len
            burst = burst[(len(burst)-N)//2:(len(burst)+N)//2] if N < len(burst) else np.pad(burst,(0,N-len(burst)))
            xt = torch.tensor(np.stack([burst.real, burst.imag], -1 if time_major else 0),
                              dtype=torch.float32, device=args.device).unsqueeze(0)
            with torch.no_grad(): out = net(xt)
            out = out[0] if isinstance(out,tuple) else out
            if isinstance(out,dict): out = out.get("gain", out)
            if out.ndim==3:             gain = torch.complex(out[0,:,0],out[0,:,1]).mean().cpu().item()*rms
            elif out.ndim==2:           gain = complex(out[0,0].cpu().item(), out[0,1].cpu().item())*rms
            else:                       gain = np.sqrt(1e-3*10**(out.cpu().item()/10))*rms
            n_full  = np.arange(len(mix_raw), dtype=np.float32)
            estimate= gain * np.exp(+1j*2*np.pi*2.0e5*n_full/args.fs).astype(np.complex64)
        else:
            est_vec = np.load(args.pred_root / mix_rel, allow_pickle=True)["x"].astype(np.complex64)
            estimate= freq_shift(est_vec[:len(mix_raw)], args.fs, 2.0e5)

        # common −200 kHz shift
        mix  = freq_shift(mix_raw,  args.fs, 2.0e5)
        est  = estimate if net else freq_shift(estimate, args.fs, 2.0e5)

        # ---------- (1) TIME DOMAIN -----------------------------------
        fig_w, ax_w = plt.subplots(figsize=(3.5,2.0))
        for sig,tag,col in zip((mix,clean,est), ("Mix I","Clean I","Est I"), colours[:3]):
            ax_w.plot(t_ax, sig.real[:n_view], label=tag, color=col)
        ax_w.grid(ls=":"); ax_w.legend(fontsize=6,loc="upper right")
        ax_w.set_xlim(0, 250)
        fig_w.tight_layout()
        fig_w.savefig(args.out_dir / f"CW{-cw:+d}_QP{qp:+d}_wave.png", dpi=args.dpi, facecolor="white")
        plt.close(fig_w)

        # ---------- (2) PSD  ------------------------------------------
        f_m,P_m = welch_psd_dbm(mix, args.fs)
        f_c,P_c = welch_psd_dbm(clean,args.fs)
        f_e,P_e = welch_psd_dbm(est, args.fs)
        mask    = np.abs(f_m)<=500
        fig_p, ax_p = plt.subplots(figsize=(3.5,2.0))
        ax_p.plot(f_m[mask], P_m[mask], label="Mix",   color=colours[0])
        ax_p.plot(f_c[mask], P_c[mask], label="Clean", color=colours[1])
        ax_p.plot(f_e[mask], P_e[mask], label="Est",   color=colours[2], ls="--", lw=0.5)
        ax_p.set(xlabel="Frequency (kHz)", ylabel="PSD (dBm/Hz)", ylim=(-150,-50))
        ax_p.grid(ls=":"); ax_p.legend(fontsize=6, loc="upper right")
        fig_p.tight_layout()
        fig_p.savefig(args.out_dir / f"CW{-cw:+d}_QP{qp:+d}_psd.png", dpi=args.dpi, facecolor="white")
        plt.close(fig_p)

        # ---------- (3) CONSTELLATION ---------------------------------
     

        plotted += 1
        print(f"[{plotted}] CW{cw:+d}  QP{qp:+d}  →  wave, psd, const")

    print(f"✓ {plotted} combinations plotted → {args.out_dir.resolve()}")

if __name__ == "__main__":
    main()








