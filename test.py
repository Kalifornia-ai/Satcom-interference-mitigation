#!/usr/bin/env python3
"""
sanity_check_dataset.py  –  Quick integrity & consistency audit
for a CW-power dataset built by build_dataset_tree.py.

The script scans every *.npz* file below a root folder, verifies the presence
of the raw burst (“x”) and its metadata (“pristine_gain”), and then prints a
short report.  With --plot it also shows two diagnostic scatter plots:

    1)  |g|      (linear)   versus folder CW power (dBm)
    2)  20·log10|g|  (dBFS) versus folder CW power (dBm)

Usage
-----
$ python sanity_check_dataset.py /path/to/dataset --plot
$ python sanity_check_dataset.py /path/to/dataset --regex "CWpwr_(-?\d+)dBm"
"""

from __future__ import annotations
from pathlib import Path
import re, argparse, json, math, sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List
try:
    from tqdm import tqdm
except ImportError:   # tqdm is purely optional
    tqdm = lambda x, **kw: x     # type: ignore


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run a quick sanity-check on every .npz in the dataset tree",
    )
    ap.add_argument("root", type=Path,
                    help="root folder created by build_dataset_tree.py")
    ap.add_argument("--expected-len", type=int, default=1000,
                    help="expected complex-burst length")
    ap.add_argument("--regex", default=r"(-?\d+)dBm",
                    help="regex (with one capture group) for CW power embedded in folder names")
    ap.add_argument("--plot", action="store_true",
                    help="show both |g| and dBFS scatter plots")
    ap.add_argument("--sample", type=int, default=0,
                    help="randomly sample N files total (0 = use every file)")
    return ap.parse_args()


# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    files: List[Path] = sorted(args.root.rglob("*.npz"))
    if not files:
        sys.exit(f"❌  No .npz files found under {args.root}")

    if args.sample > 0 and args.sample < len(files):
        rng = np.random.default_rng(0)
        files = rng.choice(files, size=args.sample, replace=False).tolist()

    pat_pwr = re.compile(args.regex)

    # Counters & collectors -------------------------------------------------
    structural_errs: List[str] = []
    len_mismatch = dtype_mismatch = 0
    xs: List[float] = []        # folder CW power (dBm)
    ys_mag: List[float] = []    # |g|   (linear)
    folder_g = {}               # folder → list[|g|]  for per-folder σ

    for f in tqdm(files, desc="Scanning"):
        try:
            z = np.load(f, allow_pickle=True)
        except Exception as e:
            structural_errs.append(f"cannot read {f}: {e}")
            continue

        if not {"x", "meta"}.issubset(z.files):
            structural_errs.append(f"missing keys in {f.relative_to(args.root)}")
            continue

        x = z["x"]
        meta = z["meta"].item()

        g = meta.get("pristine_gain")
        if g is None:
            structural_errs.append(f"no pristine_gain in {f}")
            continue

        # Shape / dtype checks ---------------------------------------------
        if len(x) != args.expected_len:
            len_mismatch += 1
        if x.dtype != np.complex64:
            dtype_mismatch += 1

        # Per-folder variance ----------------------------------------------
        folder = f.parent.name
        folder_g.setdefault(folder, []).append(abs(g))

        # Power correlation -------------------------------------------------
        m = pat_pwr.search(folder)
        if m:
            try:
                cw_dbm = int(m.group(1))
                xs.append(cw_dbm)
                ys_mag.append(abs(g))
            except ValueError:
                structural_errs.append(f"folder power not int-parsable in {folder}")

    # ---------- textual report --------------------------------------------
    print(f"\n✓ files scanned: {len(files)}\n")

    if structural_errs:
        print("❌ structural errors:")
        for e in structural_errs[:10]:
            print("   ", e)
        if len(structural_errs) > 10:
            print(f"   … {len(structural_errs)-10} more\n")
    else:
        print("✓ all files contain x + pristine_gain")

    if len_mismatch or dtype_mismatch:
        print(f"❌ len != {args.expected_len}: {len_mismatch} files  |  wrong dtype: {dtype_mismatch}\n")
    else:
        print("✓ burst length and dtype OK for all files\n")

    # σ(|g|) per folder -----------------------------------------------------
    sigmas = [np.std(20*np.log10(np.maximum(folder_arr, 1e-20)))
              for folder_arr in folder_g.values()]
    if sigmas:
        print(f"σ(|g|) per clean folder:  mean {np.mean(sigmas):.3f} dB   max {np.max(sigmas):.3f} dB\n")

    # ---------------- plotting --------------------------------------------
    if args.plot and xs:
        xs      = np.asarray(xs)
        ys_mag  = np.asarray(ys_mag) /1.4
        ys_dbfs = 20 * np.log10(np.maximum(ys_mag, 1e-20))   # dB wrt full-scale
        ys_dbm = 10 * np.log10(np.maximum((ys_mag**2)/50, 1e-40)) + 30

        r_mag  = np.corrcoef(xs, ys_mag )[0, 1]
        r_dbfs = np.corrcoef(xs, ys_dbfs)[0, 1]
        r_dbm = np.corrcoef(xs, ys_dbm)[0, 1]

        fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

        # (1) |g| linear ----------------------------------------------------
        ax[0].scatter(xs, ys_mag, s=12)
        ax[0].set_xlabel("Folder CW power (dBm)")
        ax[0].set_ylabel("|g|  (linear)")
        ax[0].set_title(f"|g|  vs  CW power   r = {r_mag:.3f}")
        ax[0].grid(True, alpha=0.3)

        # (2) |g| in dBFS ---------------------------------------------------
        xmin, xmax = xs.min(), xs.max()
        ax[1].scatter(xs, ys_dbm, s=12, c="tab:green")
        ax[1].plot([xmin, xmax], [xmin, xmax], "--", lw=1)     # 45° reference
        ax[1].set_xlabel("Folder CW power (dBm)")
        ax[1].set_ylabel("Decoded CW power (dBm)")
        ax[1].set_title(f"|g|→dBm  vs  CW power   r = {r_dbm:.3f}")
        ax[1].grid(True, alpha=0.3)

        fig.suptitle("Sanity-check: CW folder power vs decoded gain", fontsize=14)
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()


