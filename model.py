#!/usr/bin/env python3
"""
cw_power_model.py – Train a small CNN to estimate CW power (dBm)
from raw 1-k-sample CW+QPSK bursts.

Differences vs. earlier version
────────────────────────────────
• Re/Im head (2 units) but loss is MSE in dB².
• Validation loop now returns **MSE_dB²**, not RMSE.
• Loss-curve PNG shows training & validation MSE_dB².
"""

from pathlib import Path
import math, argparse, random, json
import torch, torch.nn as nn, torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

from data_loader import CWPowerDataset

# ───────── Model ───────────────────────────────────────────────────────
class BeaconPowerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.front = nn.Sequential(
            nn.Conv1d(2, 16, 9, padding=4), nn.ReLU(),
            nn.Conv1d(16, 32, 7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2),                # (Re g, Im g)
        )

    def forward(self, x):                   # (B,2,T) → (B,2)
        return self.regressor(self.front(x)).squeeze(-1)

# ───────── Helpers ─────────────────────────────────────────────────────
def vec_to_dBm(g_vec: torch.Tensor) -> torch.Tensor:
    """(B,2) Re/Im → (B,) dBm"""
    mag2 = g_vec.pow(2).sum(-1)
    return 10.0 / math.log(10) * torch.log(mag2 / 1e-3)

def freq_unshift_batch(x, f_shift, fs):
    B,_,T = x.shape; dev,dtype = x.device, x.dtype
    n = torch.arange(T, device=dev, dtype=dtype)
    mixer = torch.exp( 2j*math.pi*f_shift*n/fs ).to(torch.complex64)
    xc = torch.view_as_complex(x.permute(0,2,1).contiguous()) * mixer
    return torch.view_as_real(xc).permute(0,2,1)

def folder_balanced_split(files, seed=0, cap=0):
    rng = np.random.default_rng(seed)
    folders = np.array([f.parent.name for f in files])
    idx_all = np.arange(len(files))
    tr,val,te = [],[],[]
    for fld in np.unique(folders):
        idx = idx_all[folders==fld]; rng.shuffle(idx)
        if cap: idx = idx[:cap]
        n = len(idx); n_tr = int(.7*n); n_val = int(.15*n)
        tr += idx[:n_tr].tolist(); val += idx[n_tr:n_tr+n_val].tolist()
        te += idx[n_tr+n_val:].tolist()
    return map(np.array,(tr,val,te))

# ───────── Train / Eval loops ─────────────────────────────────────────
def train_one_epoch(model, loader, optim, dev, unshift, fs):
    model.train(); s=0.
    for x,y in loader:
        x,y=x.to(dev),y.to(dev)
        if unshift: x=freq_unshift_batch(x,2e5,fs)
        tgt = vec_to_dBm(y)
        pred= vec_to_dBm(model(x))
        loss = nn.functional.mse_loss(pred,tgt)
        optim.zero_grad(); loss.backward(); optim.step()
        s += loss.item()*x.size(0)
    return s/len(loader.dataset)          # mean MSE_dB²

@torch.no_grad()
def eval_mse(model, loader, dev, unshift, fs):
    model.eval(); s=0.; n=0
    for x,y in loader:
        x,y=x.to(dev),y.to(dev)
        if unshift: x=freq_unshift_batch(x,2e5,fs)
        tgt = vec_to_dBm(y)
        pred= vec_to_dBm(model(x))
        s += (pred-tgt).pow(2).sum().item(); n+=x.size(0)
    return s/n                              # mean MSE_dB²

# ───────── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    cudnn.benchmark=True
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("data_root",type=Path)
    ap.add_argument("--epochs",type=int,default=40)
    ap.add_argument("--batch",type=int,default=256)
    ap.add_argument("--unshift",action="store_true")
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--crop",type=int,default=1000)
    ap.add_argument("--fs",type=float,default=10e6)
    ap.add_argument("--jit-name",default="")
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--device",choices=["cuda","cpu"],
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--best-name",default="best_beacon_cnn.pt")
    ap.add_argument("--meta-name",default="cw_power_train_meta.json")
    ap.add_argument("--limit-per-folder",type=int,default=0)
    args=ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    ds=CWPowerDataset(args.data_root,crop=args.crop)
    tr_idx,va_idx,te_idx = folder_balanced_split(ds.files,args.seed,args.limit_per_folder)
    dl_tr = DataLoader(Subset(ds,tr_idx),args.batch,shuffle=True,drop_last=True,
                       num_workers=4,pin_memory=args.device=="cuda")
    dl_va = DataLoader(Subset(ds,va_idx),args.batch*4,shuffle=False,
                       num_workers=4,pin_memory=args.device=="cuda")
    dl_te = DataLoader(Subset(ds,te_idx),args.batch*4,shuffle=False,
                       num_workers=4,pin_memory=args.device=="cuda")

    print(f"train {len(tr_idx)}  val {len(va_idx)}  test {len(te_idx)}")

    model=BeaconPowerCNN().to(args.device)
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr)

    best, best_ep = float('inf'), -1
    ckpt = args.data_root/args.best_name
    tr_hist, va_hist = [], []

    for ep in range(1,args.epochs+1):
        tr_mse = train_one_epoch(model,dl_tr,optim,args.device,args.unshift,args.fs)
        va_mse = eval_mse(model,dl_va,args.device,args.unshift,args.fs)
        tr_hist.append(tr_mse); va_hist.append(va_mse)
        print(f"[E{ep:02d}] train MSE_dB² {tr_mse:.4e} | val MSE_dB² {va_mse:.4e}")
        if va_mse < best:
            best, best_ep = va_mse, ep
            torch.save(model.state_dict(), ckpt)
            print(f"  ✓ saved new best model @ epoch {ep}")

    model.load_state_dict(torch.load(ckpt,map_location=args.device))
    te_mse = eval_mse(model,dl_te,args.device,args.unshift,args.fs)
    print(f"best val MSE {best:.4e} dB²  (epoch {best_ep})")
    print(f"test MSE     {te_mse:.4e} dB²")

    meta = dict(seed=args.seed,best_val_mse=best,test_mse=te_mse,
                best_epoch=best_ep,batch=args.batch,lr=args.lr,
                limit_per_folder=args.limit_per_folder,
                train_idx=tr_idx.tolist(),val_idx=va_idx.tolist(),test_idx=te_idx.tolist())
    (args.data_root/args.meta_name).write_text(json.dumps(meta,indent=2))

    # loss curve ---------------------------------------------------------
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot(tr_hist,label="train MSE (dB²)")
    ax.plot(va_hist,label="val MSE (dB²)")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (dB²)"); ax.grid(ls=":")
    ax.legend(frameon=False); fig.tight_layout()
    loss_png=args.data_root/"loss_curve.png"
    fig.savefig(loss_png,dpi=220); plt.close(fig)
    print("✓ loss curve saved →", loss_png)

    # optional TorchScript export ---------------------------------------
    if args.jit_name:
        torch.jit.script(model.to("cpu").eval()).save(args.data_root/args.jit_name)
        print("✓ TorchScript model written to", args.data_root/args.jit_name)





