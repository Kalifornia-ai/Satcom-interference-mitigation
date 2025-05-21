#!/usr/bin/env python3
"""
build_dataset_tree.py   <measured_root>  <out_root>
                        --fs 10e6
                        [--bw 3e3] [--offset 2e5]

Minimal version – *raw mixture + pristine gain only*
====================================================
• Clean‑tone folders (e.g. `Sine_50002_-10dBm`) → one **pristine_gain** per file
  (bias‑free dot‑product on the CW‑only burst).
• Mixture folders (e.g. `Sine_50002-10dBm_QPSK_*_-20dBm`) →
  save **raw, frequency‑shifted mixture** and the matching `pristine_gain`.

Each saved `.npz` now contains **only**
```python
x         # complex64, raw 1 000‑sample burst (after coarse freq‑shift)
meta = {
   'pristine_gain' : <complex64>,   # ground‑truth channel gain
   'tone_pwr_dBm'  : <int>,         # CW level label
   'mix_file'      : '…csv',        # original CSV name
   'clean_file'    : 'clean#0003',  # CW‑only ref used
   'lag_samples'   : <int>,         # best cross‑correlation lag (debug)
   'offset_hz'     : 200 000,       # coarse shift you applied
   'fs'            : 10_000_000.0,  # sample‑rate
}
```
Nothing else (no padded `y`, no LS `complex_gain`). Perfect for training a
power‑only ML model that expects the raw burst and a pristine label.
"""

from pathlib import Path
import re, argparse, numpy as np
from scipy.signal import firwin, lfilter, fftconvolve

# ──────────────────── DSP helpers ─────────────────────────────────────────

def read_csv(p: Path) -> np.ndarray:
    """Load `<I,Q>` CSV → complex64 array."""
    a = np.loadtxt(p, delimiter=",", skiprows=1, dtype=np.float32)
    return (a[:, 0] + 1j * a[:, 1]).astype(np.complex64)


def freq_shift(x: np.ndarray, fs: float, f0: float) -> np.ndarray:
    if f0 == 0:
        return x
    n = np.arange(len(x), dtype=np.float32)
    return x * np.exp(-1j * 2 * np.pi * f0 * n / fs)


def bandpass(x: np.ndarray, fs: float, bw: float) -> np.ndarray:
    taps = firwin(65, bw / 2, fs=fs, pass_zero=True)
    return lfilter(taps, [1.0], x)


def pristine_gain(vec: np.ndarray) -> np.complex64:
    """Compute bias‑free complex gain of a CW burst already centred at DC."""
    w = np.hanning(len(vec))
    return (np.vdot(vec * w, w) / np.vdot(w, w)).astype(np.complex64)


def best_match(mix: np.ndarray, clean_bank: list, fs: float, bw: float):
    """Return (index, lag) of the clean burst that maximises correlation with mix."""
    mix_f = bandpass(mix, fs, bw)
    best_R, best_tau, best_idx = 0, 0, None
    for idx, c in enumerate(clean_bank):
        R = fftconvolve(np.conj(bandpass(c, fs, bw)[::-1]), mix_f, mode="full")
        pk = np.abs(R).max()
        if pk > best_R:
            best_R, best_tau, best_idx = pk, np.argmax(np.abs(R)) - len(c) + 1, idx
    return best_idx, best_tau

# ─────────────────── regex patterns ──────────────────────────────────────
PAT_MIX   = re.compile(r"Sine_50002[_-]?(-\d+)dBm_QPSK_[\d.]+G_(-?\d+)dBm/?$")
PAT_CLEAN = re.compile(r"Sine_50002[_-]?(-\d+)dBm/?$")

# ─────────────────── main ────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("measured_root", type=Path)
    ap.add_argument("out_root",      type=Path)
    ap.add_argument("--fs",     required=True, type=float)
    ap.add_argument("--bw",     default=3e3,   type=float)
    ap.add_argument("--offset", default=2e5,   type=float,
                    help="coarse CW shift (Hz)")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    # 1) Build clean‑tone banks & their pristine gains --------------------
    clean_bank_map, pristine_map = {}, {}
    for d in args.measured_root.iterdir():
        m = PAT_CLEAN.match(d.name)
        if not m:
            continue
        pwr = int(m.group(1))
        bank, gains = [], []
        for f in sorted(d.glob("*.csv")):
            vec = freq_shift(read_csv(f), args.fs, args.offset)  # centre CW at DC
            bank.append(vec)
            gains.append(pristine_gain(vec))
        clean_bank_map[pwr] = bank
        pristine_map[pwr]   = gains

    if not clean_bank_map:
        raise SystemExit("No clean‑tone folders found!")

    # 2) Walk mixture folders -------------------------------------------
    for d in sorted(args.measured_root.iterdir()):
        m = PAT_MIX.match(d.name)
        if not m:
            continue
        tone_pwr = int(m.group(1))
        if tone_pwr not in clean_bank_map:
            print(f"[skip] no clean CW at {tone_pwr} dBm  →  {d.name}")
            continue

        clean_bank   = clean_bank_map[tone_pwr]
        pristine_vec = pristine_map[tone_pwr]
        out_dir = args.out_root / d.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for csv in sorted(d.glob("*.csv")):
            mix = freq_shift(read_csv(csv), args.fs, args.offset)
            idx, tau = best_match(mix, clean_bank, args.fs, args.bw)
            rms_mix = np.sqrt(np.mean(np.abs(mix)**2)).astype(np.float32)

            np.savez_compressed(
                out_dir / f"{csv.stem}.npz",
                x   = mix.astype(np.complex64), 
                meta=dict(
                    rms_mix    = rms_mix,
                    pristine_gain = pristine_vec[idx],
                    offset_hz = args.offset,
                ),
            )
        print(f"[ok] {d.name}  →  {len(list(d.glob('*.csv')))} files saved")


if __name__ == "__main__":
    main()


