import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

###############################################################################
# CWPowerDataset  – now with optional *undo_shift* flag
###############################################################################

class CWPowerDataset(Dataset):
    """Dataset of raw bursts + pristine gain.

    Parameters
    ----------
    root : Path
        Root folder produced by *build_dataset_tree.py*.
    crop : int, optional
        Random crop length. 0 → use full burst. Default 1000.
    complex_out : bool, optional
        Return complex tensor instead of 2‑channel real. Default False.
    undo_shift : bool, optional
        If True, multiply every burst by exp(+j2π·offset·n/fs) to *undo* the
        200 kHz (or any) coarse shift saved in meta. Default False.
    """

    def __init__(self, root: Path, *, crop: int = 1000, complex_out: bool = False, undo_shift: bool = False):
        self.files: List[Path] = sorted(root.rglob("*.npz"))
        if not self.files:
            raise ValueError(f"No .npz files found under {root}")
        self.crop = crop
        self.complex_out = complex_out
        self.undo_shift = undo_shift

    # ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.files)

    # ----------------------------------------------------------
    def _random_window(self, x: np.ndarray) -> np.ndarray:
        if self.crop == 0 or self.crop >= len(x):
            return x
        start = np.random.randint(0, len(x) - self.crop + 1)
        return x[start:start + self.crop]

    # ----------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with np.load(self.files[idx], allow_pickle=True) as z:
            x = z["x"].astype(np.complex64)            # raw burst
            meta = z["meta"].item()
            g = meta["pristine_gain"].astype(np.complex64)

        # optional inverse frequency shift ---------------------------------
        if self.undo_shift and meta.get("offset_hz", 0) != 0:
            fs = meta["fs"]
            f0 = meta["offset_hz"]
            n = np.arange(len(x), dtype=np.float32)
            x = x * np.exp(+1j * 2 * np.pi * f0 * n / fs)

        # random crop -------------------------------------------------------
        x_win = self._random_window(x)

        # unit‑RMS normalisation -------------------------------------------
        rms = np.sqrt(np.mean(np.abs(x_win)**2))
        x_win /= rms
        g_norm = g / rms                               # keep label on same scale

        # to tensor ---------------------------------------------------------
        if self.complex_out:
            x_t = torch.view_as_real(torch.from_numpy(x_win)).float()
            x_t = torch.view_as_complex(x_t)
        else:
            x_t = torch.from_numpy(np.stack([x_win.real, x_win.imag], 0))

        target = torch.tensor([g_norm.real, g_norm.imag], dtype=torch.float32)
        return x_t, target

###############################################################################
# Helper factory (unchanged except for undo_shift passthrough)
###############################################################################

def make_loaders(root: Path, *, batch: int = 256, crop: int = 1000, val_frac=0.1, undo_shift=False, **dl_kwargs):
    files = sorted(root.rglob("*.npz"))
    train_files = [f for i, f in enumerate(files) if (i % int(1/val_frac))]
    val_files   = [f for i, f in enumerate(files) if not (i % int(1/val_frac))]

    ds_train = CWPowerDataset(root, crop=crop, undo_shift=undo_shift)
    ds_val   = CWPowerDataset(root, crop=crop, undo_shift=undo_shift)
    ds_train.files = train_files
    ds_val.files   = val_files

    dl_train = DataLoader(ds_train, batch_size=batch, shuffle=True, drop_last=True, **dl_kwargs)
    dl_val   = DataLoader(ds_val,   batch_size=batch * 4, shuffle=False, **dl_kwargs)
    return dl_train, dl_val

###############################################################################
# CLI sanity test
###############################################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--crop", type=int, default=1000)
    parser.add_argument("--undo-shift", action="store_true")
    args = parser.parse_args()

    ds = CWPowerDataset(args.root, crop=args.crop, undo_shift=args.undo_shift)
    x, y = ds[0]
    print(np.shape(y))
    print("x", x.shape, x.dtype, "| target", y)
