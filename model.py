"""cw_power_model.py – Train a small CNN to estimate **CW power (dBm)**
from raw 1 000‑sample CW+QPSK bursts stored by the *minimal* dataset script.

The script now supports **explicit test‑set folders**: pass
```
  --test-folders FOLDER1 FOLDER2 ...
```
Every file whose parent‑directory name matches one of those strings is locked
into the *held‑out* **test** split; the remainder are split 90 / 10 into
train/val with a per‑folder shuffle (to avoid leakage).

Phase is ignored because each burst’s absolute phase is random; the network
predicts a single scalar
```
P̂_dBm = 10 · log10(|g|² / 1 mW)
```

Usage
-----
$ python cw_power_model.py  /path/to/dataset  \
        --epochs 40 --batch 256 --lr 3e-4 \
        --test-folders Sine_50002-20dBm_QPSK_... Sine_50002-25dBm_QPSK_...

Outputs
-------
* best_beacon_cnn.pt  – model weights (saved in dataset root)
* console log of train loss (dB²), val RMSE (dB) each epoch, and final test RMSE
"""

from pathlib import Path
import math, argparse, random, json, torch, torch.nn as nn, torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
import numpy as np

from data_loader import CWPowerDataset

dBm_factor = 10.0 / math.log(10)
###############################################################################
# Model
###############################################################################

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
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.regressor(self.front(x)).squeeze(-1)

###############################################################################
# Helpers
###############################################################################

def gain_to_dBm(g_reim: torch.Tensor) -> torch.Tensor:
    power_lin = g_reim.pow(2).sum(dim=-1)
    return dBm_factor * torch.log(power_lin / 1e-3)


def split_indices(files, test_folders, val_frac=0.2, seed=0):
    """Return train/val/test index lists with folder‑balanced val."""
    rng = np.random.default_rng(seed)
    folders = np.array([f.parent.name for f in files])
    idx_all = np.arange(len(files))

    test_mask = np.isin(folders, test_folders)
    val_mask  = np.zeros(len(files), dtype=bool)

    for folder in np.unique(folders[~test_mask]):
        idx_f = idx_all[(folders == folder) & (~test_mask)]
        rng.shuffle(idx_f)
        n_val = max(1, int(val_frac * len(idx_f)))
        val_mask[idx_f[:n_val]] = True

    train_mask = ~(test_mask | val_mask)
    return idx_all[train_mask], idx_all[val_mask], idx_all[test_mask]

###############################################################################
# Train / eval loops
###############################################################################

def train_one_epoch(model, loader, optim):
    model.train(); acc_loss = 0.0
    for x, y in loader:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        target = gain_to_dBm(y)
        optim.zero_grad()
        pred = model(x)
        loss = nn.functional.mse_loss(pred, target)
        loss.backward(); optim.step()
        acc_loss += loss.item() * x.size(0)
    return acc_loss / len(loader.dataset)


def eval_rmse(model, loader):
    model.eval(); se, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            target = gain_to_dBm(y)
            se += (model(x) - target).pow(2).sum().item(); n += x.size(0)
    return math.sqrt(se / n)

###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    cudnn.benchmark = True

    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("data_root", type=Path)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch",  type=int, default=256)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--crop",   type=int, default=1000)
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--test-folders", nargs="*", default=[],
                    help="folder names reserved for the test split")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # Build dataset & deterministic splits ----------------------------------
    ds_full = CWPowerDataset(args.data_root, crop=args.crop)
    train_idx, val_idx, test_idx = split_indices(ds_full.files, args.test_folders, seed=args.seed)

    ds_train = Subset(ds_full, train_idx)
    ds_val   = Subset(ds_full, val_idx)
    ds_test  = Subset(ds_full, test_idx)

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, drop_last=True,
                          num_workers=4, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch*4, shuffle=False,
                          num_workers=4, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch*4, shuffle=False,
                          num_workers=4, pin_memory=True)

    print(f"train {len(ds_train)}  val {len(ds_val)}  test {len(ds_test)}")

    # Model & optimiser ------------------------------------------------------
    model = BeaconPowerCNN().cuda()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_rmse, best_ep = float('inf'), -1
    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, dl_train, optim)
        val_rmse = eval_rmse(model, dl_val)
        print(f"[E{ep:02d}] train MSE_dB² {tr_loss:.4e} | val RMSE_dB {val_rmse:.3f}")
        if val_rmse < best_rmse:
            best_rmse, best_ep = val_rmse, ep
            torch.save(model.state_dict(), args.data_root / "best_beacon_cnn.pt")
            print(f"  ✓ saved new best model @ epoch {ep}")

    # Final test -------------------------------------------------------------
    model.load_state_dict(torch.load(args.data_root / "best_beacon_cnn.pt"))
    test_rmse = eval_rmse(model, dl_test)
    print(f"best val RMSE {best_rmse:.3f} dB  (epoch {best_ep})")
    print(f"test RMSE     {test_rmse:.3f} dB")

    # Save split metadata for reproducibility
    meta = dict(
        seed=args.seed,
        best_val_rmse=best_rmse,
        test_rmse=test_rmse,
        test_folders=args.test_folders,
        train_idx=train_idx.tolist(),
        val_idx=val_idx.tolist(),
        test_idx=test_idx.tolist(),
    )
    (args.data_root / "cw_power_train_meta.json").write_text(json.dumps(meta, indent=2))


