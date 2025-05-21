import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

###############################################################################
# Dataset that matches the *minimal* .npz produced by the updated builder
# ----------------------------------------------------------------------
# Each .npz contains:
#   x    : complex64 (1000,)           – raw burst, already freq‑shifted
#   meta : {
#            'pristine_gain': complex64,
#            'tone_pwr_dBm' : int,
#            ...
#          }
###############################################################################

class CWPowerDataset(Dataset):
    """Torch `Dataset` for raw‑burst + pristine‑gain files.

    Args
    -----
    root : Path to the tree built by *build_dataset_tree.py*.
    crop : Number of samples to slice out of each burst.  If `crop == 0`
           or equals the burst length (1000), the full burst is returned.
    complex_out : If True, returns complex tensors of shape (N,) instead of
                  stacked (2, N) real/imag channels.  Useful for complex
                  convolution layers.
    """

    def __init__(self, root: Path, *, crop: int = 1000, complex_out: bool = False):
        self.files: List[Path] = sorted(root.rglob("*.npz"))
        if not self.files:
            raise ValueError(f"No .npz files found under {root}")
        self.crop = crop
        self.complex_out = complex_out

    # ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.files)

    # ----------------------------------------------------------
    def _random_window(self, x: np.ndarray) -> np.ndarray:
        """Return a crop‑length slice (wrap if x shorter)."""
        if self.crop == 0 or self.crop >= len(x):
            return x
        start = np.random.randint(0, len(x) - self.crop + 1)
        return x[start : start + self.crop]

    # ----------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with np.load(self.files[idx], allow_pickle=True) as z:
            x_raw = z["x"].astype(np.complex64)              # (1000,)
            meta  = z["meta"].item()
            g     = meta["pristine_gain"].astype(np.complex64)

        x_win = self._random_window(x_raw)

        # normalise window RMS → 1  (common in earlier pipeline)
        #x_win = x_win / np.sqrt(np.mean(np.abs(x_win) ** 2))

        rms = np.sqrt(np.mean(np.abs(x_win)**2))
        x_win /= rms

        # convert to torch tensor
        if self.complex_out:
            x_t = torch.view_as_real(torch.from_numpy(x_win)).float()
            x_t = torch.view_as_complex(x_t)                 # dtype: complex64
        else:
            x_t = torch.from_numpy(np.stack([x_win.real, x_win.imag], 0))

       
        g = meta["pristine_gain"] / meta["rms_mix"] 
        target = np.array([ g.real,
                    g.imag ], np.float32)
        return x_t, target

###############################################################################
# Convenience factory: split train/val by modulo index
###############################################################################

def make_loaders(root: Path, *, batch: int = 256, crop: int = 1000, val_frac=0.1, **dl_kwargs):
    files = sorted(root.rglob("*.npz"))
    train_files = [f for i, f in enumerate(files) if (i % int(1/val_frac))]
    val_files   = [f for i, f in enumerate(files) if not (i % int(1/val_frac))]

    ds_train = CWPowerDataset(root, crop=crop)
    ds_val   = CWPowerDataset(root, crop=crop)
    ds_train.files = train_files
    ds_val.files   = val_files

    dl_train = DataLoader(ds_train, batch_size=batch, shuffle=True, drop_last=True, **dl_kwargs)
    dl_val   = DataLoader(ds_val,   batch_size=batch * 4, shuffle=False, **dl_kwargs)
    return dl_train, dl_val

###############################################################################
# Example usage (run as `python dataset_loader.py /path/to/dataset`)
###############################################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--crop", type=int, default=1000)
    args = parser.parse_args()

    dl_train, dl_val = make_loaders(args.root, batch=args.batch, crop=args.crop, num_workers=2)
    x, y = next(iter(dl_train))
    print("x", x.shape, x.dtype)
    print("y", y.shape, y.dtype)
