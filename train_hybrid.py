#!/usr/bin/env python3
"""
train_hybrid_beacon.py – Train HybridBeaconEstimator with the same
folder-balanced split and dB-domain loss used in cw_power_model.py.

Example
-------
python train_hybrid_beacon.py ./dataset \
       --epochs 60 --batch 256 --lr 3e-4 --device cuda \
       --best-name hybrid_best.pt --meta-name hybrid_meta.json \
       --limit-per-folder 0
"""
from pathlib import Path
import math, json, argparse, random
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset

from data_loader import CWPowerDataset
from resmodel import HybridBeaconEstimator   # your model file

dBm_factor = 10.0 / math.log(10)

# ────────── helpers ────────────────────────────────────────────────────
def gain_to_dBm(g_reim: torch.Tensor) -> torch.Tensor:
    """convert shape (...,2) [re,im] → dBm scalar tensor"""
    power_lin = g_reim.pow(2).sum(-1)           # |g|²  (W @ 1 Ω)
    return dBm_factor * torch.log(power_lin / 1e-3)

def folder_balanced_split(files, seed=0, limit_per_folder=0):
    """Return 70/15/15 split indices with optional cap per folder."""
    rng = np.random.default_rng(seed)
    folders = np.array([f.parent.name for f in files])
    idx_all = np.arange(len(files))

    tr,val,te = [],[],[]
    for folder in np.unique(folders):
        idx_f = idx_all[folders == folder]
        rng.shuffle(idx_f)
        if limit_per_folder > 0:
            idx_f = idx_f[:limit_per_folder]
        n = len(idx_f)
        n_tr = int(0.70*n); n_val = int(0.15*n)
        tr.extend(idx_f[:n_tr])
        val.extend(idx_f[n_tr:n_tr+n_val])
        te.extend(idx_f[n_tr+n_val:])
    return map(np.array,(tr,val,te))

# ────────── train / val loops ──────────────────────────────────────────
def run_epoch(model, loader, optim, train: bool, device):
    model.train() if train else model.eval()
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:                      # x:(B,N,2)  y:(B,2)
            x, y = x.to(device), y.to(device)
            pred = model(x)["gain"]              # (B,2)
            loss = nn.functional.mse_loss(gain_to_dBm(pred), gain_to_dBm(y))
            if train:
                optim.zero_grad(); loss.backward(); optim.step()
            total += loss.item()*x.size(0); n += x.size(0)
    return total / n                             # mean MSE in dB²

# ────────── main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("data_root", type=Path)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch",  type=int, default=256)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--crop",   type=int,   default=1000)
    ap.add_argument("--seed",   type=int,   default=0)
    ap.add_argument("--device", choices=["cuda","cpu"],
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--best-name", default="hybrid_best.pt")
    ap.add_argument("--meta-name", default="hybrid_meta.json")
    ap.add_argument("--limit-per-folder", type=int, default=0,
                    help="cap N bursts per folder (0 = use all)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # dataset & balanced split -------------------------------------------
    ds_full = CWPowerDataset(args.data_root, crop=args.crop)
    tr_idx, va_idx, te_idx = folder_balanced_split(ds_full.files, seed=args.seed,
                                                   limit_per_folder=args.limit_per_folder)
    ds_tr, ds_va, ds_te = (Subset(ds_full, idx) for idx in (tr_idx,va_idx,te_idx))
    pin = args.device=="cuda"
    dl_tr = DataLoader(ds_tr, batch_size=args.batch,   shuffle=True,  drop_last=True,
                       num_workers=4, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=args.batch*4, shuffle=False,
                       num_workers=4, pin_memory=pin)
    dl_te = DataLoader(ds_te, batch_size=args.batch*4, shuffle=False,
                       num_workers=4, pin_memory=pin)
    print(f"train {len(ds_tr)}  val {len(ds_va)}  test {len(ds_te)}")

    # model / optimiser ---------------------------------------------------
    model = HybridBeaconEstimator().to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best, best_ep = float("inf"), -1
    ckpt = args.data_root / args.best_name
    tr_curve, va_curve = [], []

    # loop ---------------------------------------------------------------
    for ep in range(1, args.epochs+1):
        tr = run_epoch(model, dl_tr, optim, True,  args.device)
        va = run_epoch(model, dl_va, optim, False, args.device)
        tr_curve.append(tr); va_curve.append(va)
        print(f"[E{ep:02d}] train MSE_dB² {tr:.4e} | val MSE_dB² {va:.4e}")
        if va < best:
            best, best_ep = va, ep
            torch.save(model.state_dict(), ckpt)
            print("   ✓ new best model")

    # test ---------------------------------------------------------------
    model.load_state_dict(torch.load(ckpt, map_location=args.device))
    te = run_epoch(model, dl_te, optim, False, args.device)
    print(f"best val {best:.4e} dB²  (epoch {best_ep})")
    print(f"test MSE {te:.4e} dB²")

    # loss curve ---------------------------------------------------------
    import matplotlib.pyplot as plt
    e = np.arange(1,len(tr_curve)+1)
    plt.plot(e, tr_curve,label="train"); plt.plot(e, va_curve,label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE (dB²)"); plt.grid(ls=':')
    plt.legend(frameon=False); plt.tight_layout()
    plt.savefig(args.data_root/"hybrid_loss_curve.png", dpi=220); plt.close()

    # metadata -----------------------------------------------------------
    meta = dict(seed=args.seed, best_val_mse_dB2=best, test_mse_dB2=te,
                best_epoch=best_ep, batch=args.batch, lr=args.lr,
                limit_per_folder=args.limit_per_folder,
                train_idx=tr_idx.tolist(), val_idx=va_idx.tolist(),
                test_idx=te_idx.tolist())
    (args.data_root/args.meta_name).write_text(json.dumps(meta,indent=2))

