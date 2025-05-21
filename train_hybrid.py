#!/usr/bin/env python3
"""train_hybrid_beacon.py – Train the *HybridBeaconEstimator* (Res‑CNN + LSTM)
against pristine complex‑gain labels.

Uses the same dataset tree produced by *build_dataset_tree.py* and the same
folder‑balanced 70 / 15 / 15 split logic as *cw_power_model.py*.

Key options
-----------
--limit-per-folder N   cap samples per folder (quick experiments)
--best-name NAME.pt    custom checkpoint filename  (default hybrid_best.pt)
--meta-name NAME.json  custom JSON metadata file   (default hybrid_meta.json)
--device cpu|cuda      run on CPU if CUDA not desired

Example
-------
```bash
python train_hybrid_beacon.py ./dataset --epochs 60 --batch 64 --device cuda \
       --best-name hybrid_lstm.pt --meta-name hybrid_meta.json
```"""

from pathlib import Path
import math, argparse, random, json
import numpy as np
import torch, torch.nn as nn, torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset

from dataset_loader import CWPowerDataset    # returns (2,N) real‑imag + pristine_gain
from hybrid import HybridBeaconEstimator     # the model pasted earlier

###############################################################################
# Helpers
###############################################################################

def folder_balanced_split(files, seed=0, limit=0):
    rng = np.random.default_rng(seed)
    folders = np.array([f.parent.name for f in files])
    idx_all = np.arange(len(files))
    tr, va, te = [], [], []
    for folder in np.unique(folders):
        idx_f = idx_all[folders == folder]
        rng.shuffle(idx_f)
        if limit:
            idx_f = idx_f[:limit]
        n = len(idx_f)
        n_tr = int(0.70 * n); n_va = int(0.15 * n)
        tr.extend(idx_f[:n_tr])
        va.extend(idx_f[n_tr:n_tr+n_va])
        te.extend(idx_f[n_tr+n_va:])
    return map(np.array, (tr, va, te))

###############################################################################
# Loss & metric
###############################################################################

def mse_complex(pred, target):
    """MSE over (Re,Im) pairs."""
    return nn.functional.mse_loss(pred, target)

###############################################################################
# Training loops
###############################################################################

def train_epoch(model, loader, opt, device):
    model.train(); total = 0.0
    for x, y in loader:               # x (2,N) real‑imag, y (2,)
        # reshape to (B,N,2)
        x = x.to(device).permute(0, 2, 1)  # (B,2,N)->(B,N,2)
        y = y.to(device)
        opt.zero_grad()
        pred_dict = model(x)
        loss = mse_complex(pred_dict["gain"], y)
        loss.backward(); opt.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def eval_rmse(model, loader, device):
    model.eval(); se, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).permute(0, 2, 1)
            y = y.to(device)
            pred = model(x)["gain"]
            se += (pred - y).pow(2).sum(dim=-1).sum().item(); n += x.size(0)
    rmse = math.sqrt(se / n)
    return rmse

###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    cudnn.benchmark = True

    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("data_root", type=Path)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch",  type=int, default=128)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--crop",   type=int, default=1000)
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--best-name", default="hybrid_best.pt")
    ap.add_argument("--meta-name", default="hybrid_meta.json")
    ap.add_argument("--limit-per-folder", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # Dataset setup --------------------------------------------------------
    ds_full = CWPowerDataset(args.data_root, crop=args.crop)
    tr_idx, va_idx, te_idx = folder_balanced_split(ds_full.files, seed=args.seed,
                                                   limit=args.limit_per_folder)
    ds_tr = Subset(ds_full, tr_idx)
    ds_va = Subset(ds_full, va_idx)
    ds_te = Subset(ds_full, te_idx)

    pin = args.device == "cuda"
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, drop_last=True,
                       num_workers=4, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=args.batch*2, shuffle=False,
                       num_workers=4, pin_memory=pin)
    dl_te = DataLoader(ds_te, batch_size=args.batch*2, shuffle=False,
                       num_workers=4, pin_memory=pin)

    print(f"train {len(ds_tr)}  val {len(ds_va)}  test {len(ds_te)}")

    # Model & optimiser ----------------------------------------------------
    model = HybridBeaconEstimator().to(args.device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_rmse, best_ep = float('inf'), -1
    ckpt_path = args.data_root / args.best_name

    for ep in range(1, args.epochs+1):
        tr_loss = train_epoch(model, dl_tr, opt, args.device)
        val_rmse = eval_rmse(model, dl_va, args.device)
        print(f"[E{ep:02d}] train MSE {tr_loss:.4e} | val RMSE {val_rmse:.4f}")
        if val_rmse < best_rmse:
            best_rmse, best_ep = val_rmse, ep
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ saved new best @ epoch {ep}")

    # Final test -----------------------------------------------------------
    model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
    test_rmse = eval_rmse(model, dl_te, args.device)
    print(f"best val RMSE {best_rmse:.4f}  (epoch {best_ep})")
    print(f"test RMSE     {test_rmse:.4f}")

    meta = dict(
        seed=args.seed,
        epochs=args.epochs,
        best_val_rmse=best_rmse,
        test_rmse=test_rmse,
        best_epoch=best_ep,
        limit_per_folder=args.limit_per_folder,
        train_idx=tr_idx.tolist(),
        val_idx=va_idx.tolist(),
        test_idx=te_idx.tolist()
    )
    (args.data_root / args.meta_name).write_text(json.dumps(meta, indent=2))