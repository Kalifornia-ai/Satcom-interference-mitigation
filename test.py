#!/usr/bin/env python3
"""Quick sanity‑check script for the CW‑power dataset.

Runs five lightweight tests and prints a short report. Optionally produces a
scatter plot of |g| versus folder CW power for a visual inspection.

Usage
-----
$ python sanity_check_dataset.py /path/to/dataset [--plot]
"""

from pathlib import Path
import re, argparse, json, math
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
PAT_PWR = re.compile(r"-?(\d+)dBm")

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("root", type=Path, help="root folder created by build_dataset_tree.py")
    ap.add_argument("--expected-len", type=int, default=1000, help="expected burst length")
    ap.add_argument("--plot", action="store_true", help="show |g| vs CW‑power scatter")
    args = ap.parse_args()

    files = sorted(args.root.rglob("*.npz"))
    if not files:
        raise SystemExit("❌  No .npz files found under", args.root)

    errs = []
    lens_wrong, dtype_wrong = 0, 0
    xs, ys = [], []               # for correlation plot
    sigmas = []                   # per‑folder σ(|g|) dB

    # organise by clean folder for variance test
    folder_g = {}

    for f in files:
        z = np.load(f, allow_pickle=True)
        if not {"x", "meta"}.issubset(z.files):
            errs.append(f"missing keys in {f.relative_to(args.root)}")
            continue

        x = z["x"]
        meta = z["meta"].item()
        g = meta.get("pristine_gain")
        if g is None:
            errs.append(f"no pristine_gain in {f}")
            continue

        if len(x) != args.expected_len:
            lens_wrong += 1
        if x.dtype != np.complex64:
            dtype_wrong += 1

        folder = f.parent.name
        folder_g.setdefault(folder, []).append(g)

        # power correlation
        m = PAT_PWR.search(folder)
        if m:
            cw_dbm = -int(m.group(1))
            xs.append(cw_dbm)
            ys.append(20*np.log10(abs(g)))

    # ------------ print results ------------------------------------------
    print("✓ files scanned:", len(files))
    if errs:
        print("❌ structural errors:")
        for e in errs[:10]:
            print(" ", e)
        if len(errs) > 10:
            print("  …", len(errs)-10, "more")
    else:
        print("✓ all files contain x + pristine_gain")

    if lens_wrong or dtype_wrong:
        print(f"❌ len!= {args.expected_len}: {lens_wrong} files | wrong dtype: {dtype_wrong}")
    else:
        print("✓ burst length and dtype OK for all files")

    # power correlation coefficient
    if xs:
        r = np.corrcoef(xs, ys)[0,1]
        print(f"|g| vs folder CW dBm correlation r = {r:.4f}")
        if args.plot:
            plt.scatter(xs, ys, s=8)
            plt.xlabel('Folder CW power (dBm)')
            plt.ylabel('|pristine_gain| (dB)')
            plt.title(f'Correlation r={r:.3f}')
            plt.show()

    # per‑folder variance
    for folder, arr in folder_g.items():
        sigma = np.std(20*np.log10(np.abs(arr)))
        sigmas.append(sigma)
    if sigmas:
        print(f"σ(|g|) per clean folder: mean {np.mean(sigmas):.3f} dB  max {np.max(sigmas):.3f} dB")
    print("Done.")

if __name__ == "__main__":
    main()
