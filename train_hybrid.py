#!/usr/bin/env python3
"""
train_hybrid_beacon.py – train HybridBeaconEstimator to predict complex CW gain.

Example
-------
python train_hybrid_beacon.py ./dataset \
        --epochs 60 --batch 128 --lr 3e-4 --device cuda \
        --best-name hybrid_best.pt --meta hybrid_meta.json
"""
from pathlib import Path
import math, json, argparse, random, numpy as np
import torch, torch.nn as nn, torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset

from data_loader import CWPowerDataset
from hybrid_beacon_estimator import HybridBeaconEstimator   # <- put your model here

dBm_factor = 10.0 / math.log(10)

# ───────────────────────── helper ──────────────────────────
def gain_to_dBm(g_reim: torch.Tensor) -> torch.Tensor:
    """convert (B,2) real-imag gain → dBm scalar"""
    p = g_reim.pow(2).sum(-1)           # |g|²  (linear W @ 1 Ω)
    return dBm_factor * torch.log(p / 1e-3)

# ───────────────────────── train / val ─────────────────────
def run_epoch(model, loader, optim, device, train=True):
    model.train() if train else model.eval()
    running, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)      # x:(B,N,2)  y:(B,2)
            pred = model(x)["gain"]                # (B,2)
            loss = nn.functional.mse_loss(pred, y)
            if train:
                optim.zero_grad(); loss.backward(); optim.step()
            running += loss.item() * x.size(0); n += x.size(0)
    return running / n                             # mean MSE on I/Q

# ───────────────────────── main ────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("data_root", type=Path)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch",  type=int, default=16)
    ap.add_argument("--lr",     type=float, default=2e-4)
    ap.add_argument("--crop",   type=int, default=1000)
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--device", choices=["cuda","cpu"],
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--best-name", default="hybrid_best.pt")
    ap.add_argument("--meta",      default="hybrid_meta.json")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # dataset split taken from the original JSON -------------------------
    ds_full = CWPowerDataset(args.data_root, crop=args.crop)
    meta_json = json.loads((args.data_root / "cw_power_train_meta.json").read_text())
    train_idx = np.array(meta_json["train_idx"]); val_idx  = np.array(meta_json["val_idx"])
    test_idx  = np.array(meta_json["test_idx"])

    ds_train, ds_val, ds_test = (Subset(ds_full, idx) for idx in (train_idx,val_idx,test_idx))
    pin = args.device=="cuda"
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, drop_last=True,
                          num_workers=4, pin_memory=pin)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch*4, shuffle=False,
                          num_workers=4, pin_memory=pin)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch*4, shuffle=False,
                          num_workers=4, pin_memory=pin)

    # model / optimiser ---------------------------------------------------
    model = HybridBeaconEstimator().to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val, best_ep = float("inf"), -1
    ckpt = args.data_root / args.best_name
    train_curve, val_curve = [], []

    # training loop -------------------------------------------------------
    for ep in range(1, args.epochs+1):
        tr = run_epoch(model, dl_train, optim, args.device, train=True)
        va = run_epoch(model, dl_val,   optim, args.device, train=False)
        train_curve.append(tr); val_curve.append(va)
        print(f"[E{ep:02d}] train MSE_IQ {tr:.4e} | val MSE_IQ {va:.4e}")
        if va < best_val:
            best_val, best_ep = va, ep
            torch.save(model.state_dict(), ckpt)
            print("   ✓ new best")

    # test ---------------------------------------------------------------
    model.load_state_dict(torch.load(ckpt, map_location=args.device))
    test_mse = run_epoch(model, dl_test, optim, args.device, train=False)
    print(f"best val MSE {best_val:.4e} (epoch {best_ep})")
    print(f"test MSE     {test_mse:.4e}")

    # quick loss-curve figure -------------------------------------------
    import matplotlib.pyplot as plt
    e = np.arange(1, len(train_curve)+1)
    plt.plot(e, train_curve, label="train"); plt.plot(e, val_curve, label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE on I/Q"); plt.legend(); plt.grid(ls=":")
    plt.tight_layout(); plt.savefig(args.data_root/"hybrid_loss_curve.png", dpi=200)
    plt.close()

    # metadata -----------------------------------------------------------
    meta = dict(seed=args.seed, best_val_mse=best_val, test_mse=test_mse,
                best_epoch=best_ep, batch=args.batch, lr=args.lr)
    (args.data_root/args.meta).write_text(json.dumps(meta, indent=2))
