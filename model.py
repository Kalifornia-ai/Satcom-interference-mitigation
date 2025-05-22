"""cw_power_model.py – Train a small CNN to estimate **CW power (dBm)**
from raw 1‑k‑sample CW+QPSK bursts.

Updates
=======
1. **70 / 15 / 15 split inside every folder** (no leakage).
2. **--best-name** and **--meta-name** flags let you choose the
   checkpoint filename and the JSON metadata filename.
3. **--limit-per-folder N** keeps only the first *N* shuffled files from each
   folder before making the split – handy when you want a quick run on a
   subset.

Example
-------
```bash
python cw_power_model.py ./dataset \
    --epochs 30 --batch 128 --lr 1e-3 --device cpu \
    --best-name cnn_cpu.pt --meta-name run_cpu.json \
    --limit-per-folder 500
```
"""

from pathlib import Path
import math, argparse, random, json
import torch, torch.nn as nn, torch.backends.cudnn as cudnn
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
            nn.Linear(32, 1),  # dBm
        )

    def forward(self, x):
        return self.regressor(self.front(x)).squeeze(-1)

###############################################################################
# Helpers
###############################################################################

def gain_to_dBm(g_reim: torch.Tensor) -> torch.Tensor:
    power_lin = g_reim.pow(2).sum(dim=-1)
    return dBm_factor * torch.log(power_lin / 1e-3)


def freq_unshift_batch(x: torch.Tensor,
                       f_shift: float,
                       fs: float) -> torch.Tensor:
    """
    Mix a batch of [B, 2, T] real-imag bursts by –f_shift Hz.

    Parameters
    ----------
    x : Tensor
        Shape [B, 2, T], dtype float32/float64.
    f_shift : float
        Frequency to remove (positive Hz).
    fs : float
        Sample-rate in Hz.
    """
    B, _, T = x.shape
    device, dtype = x.device, x.dtype

    n = torch.arange(T, device=device, dtype=dtype)
    phase = 2.0 * math.pi * f_shift * n / fs          # –2πft/fs  (rad)
    mixer = torch.exp(phase * (1j)).to(dtype=torch.complex64)  # [T] complex

    # reshape mixer to [1, T] then broadcast over batch
    x_c = torch.view_as_complex(x.permute(0, 2, 1).contiguous())  # [B, T]
    x_c = x_c * mixer                                             # mix down
    x_shifted = torch.view_as_real(x_c).permute(0, 2, 1)          # back to [B, 2, T]
    return x_shifted


def folder_balanced_split(files, seed=0, limit_per_folder=0):
    """Return 70/15/15 split indices with optional cap per folder."""
    rng = np.random.default_rng(seed)
    folders = np.array([f.parent.name for f in files])
    idx_all = np.arange(len(files))

    train_idx, val_idx, test_idx = [], [], []
    for folder in np.unique(folders):
        idx_f = idx_all[folders == folder]
        rng.shuffle(idx_f)
        if limit_per_folder > 0:
            idx_f = idx_f[:limit_per_folder]
        n = len(idx_f)
        n_train = int(0.70 * n)
        n_val   = int(0.15 * n)
        train_idx.extend(idx_f[:n_train])
        val_idx.extend(idx_f[n_train:n_train + n_val])
        test_idx.extend(idx_f[n_train + n_val:])
    return map(np.array, (train_idx, val_idx, test_idx))

###############################################################################
# Train / eval loops
###############################################################################

def train_one_epoch(model, loader, optim, device,   do_unshift: bool, fs: float):
    model.train(); total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if do_unshift:
            x = freq_unshift_batch(x, 2.0e5, fs)     # –200 kHz
        target = gain_to_dBm(y)
        optim.zero_grad()
        loss = nn.functional.mse_loss(model(x), target)
        loss.backward(); optim.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def eval_rmse(model, loader, device, do_unshift: bool, fs: float):
    model.eval(); se, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if do_unshift:
                x = freq_unshift_batch(x, 2.0e5, fs)
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
    ap.add_argument("--unshift", action="store_true",
                help="frequency-unshift each burst by –200 kHz before training")
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--crop",   type=int, default=1000)
    ap.add_argument("--fs", type=float, default=10e6,
                help="sample-rate of the raw I/Q (Hz) – needed for the mixer")
    ap.add_argument("--jit-name", default="",
               help="TorchScript filename to write (empty = don’t export)")
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--best-name", default="best_beacon_cnn.pt", help="checkpoint filename")
    ap.add_argument("--meta-name", default="cw_power_train_meta.json", help="metadata JSON filename")
    ap.add_argument("--limit-per-folder", type=int, default=0,
                    help="cap number of samples per folder (0 = no cap)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # Dataset --------------------------------------------------------------
    ds_full = CWPowerDataset(args.data_root, crop=args.crop)
    train_idx, val_idx, test_idx = folder_balanced_split(ds_full.files, seed=args.seed,
                                                         limit_per_folder=args.limit_per_folder)
    ds_train = Subset(ds_full, train_idx)
    ds_val   = Subset(ds_full, val_idx)
    ds_test  = Subset(ds_full, test_idx)

    pin = args.device == "cuda"
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, drop_last=True,
                          num_workers=4, pin_memory=pin)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch*4, shuffle=False,
                          num_workers=4, pin_memory=pin)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch*4, shuffle=False,
                          num_workers=4, pin_memory=pin)

    print(f"train {len(ds_train)}  val {len(ds_val)}  test {len(ds_test)}")

    # Model & optimiser ----------------------------------------------------
    model = BeaconPowerCNN().to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_rmse = float('inf'); best_ep = -1
    ckpt_path = args.data_root / args.best_name

    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, dl_train, optim, args.device,  do_unshift=args.unshift, fs=args.fs)
        val_rmse = eval_rmse(model, dl_val, args.device,  do_unshift=args.unshift, fs=args.fs)
        print(f"[E{ep:02d}] train MSE_dB² {tr_loss:.4e} | val RMSE_dB {val_rmse:.3f}")
        if val_rmse < best_rmse:
            best_rmse, best_ep = val_rmse, ep
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ saved new best model @ epoch {ep}")

    # Final test -----------------------------------------------------------
    model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
    test_rmse = eval_rmse(model, dl_test, args.device,  do_unshift=args.unshift, fs=args.fs)
    print(f"best val RMSE {best_rmse:.3f} dB  (epoch {best_ep})")
    print(f"test RMSE     {test_rmse:.3f} dB")

    # Metadata -------------------------------------------------------------
    meta = dict(
        seed=args.seed,
        best_val_rmse=best_rmse,
        test_rmse=test_rmse,
        best_epoch=best_ep,
        batch=args.batch,
        lr=args.lr,
        limit_per_folder=args.limit_per_folder,
        train_idx=train_idx.tolist(),
        val_idx=val_idx.tolist(),
        test_idx=test_idx.tolist(),
    )
    (args.data_root / args.meta_name).write_text(json.dumps(meta, indent=2))

    # Optional TorchScript export -----------------------------------------
    if args.jit_name:
        script_path = args.data_root / args.jit_name
        model_cpu = model.to("cpu").eval()
        scripted  = torch.jit.script(model_cpu)        # robust to variable batch
        scripted.save(script_path)
        print(f"✓ TorchScript model written to {script_path}")




