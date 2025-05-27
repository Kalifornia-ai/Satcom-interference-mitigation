import torch, json, os
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import sys, argparse
from torch.amp import autocast, GradScaler
from pathlib import Path
from typing import List, Dict, Any, Tuple

def parse_args():
    parser = argparse.ArgumentParser(description="LSTM Single-Source Training")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--mix_path", type=str, default="../ExperimentalDNN/data/Measured Data/newcapture/Mix/", help="Path for mixture")
    parser.add_argument("--sig_path", type=str, default="../ExperimentalDNN/data/Measured Data/newcapture/CW/")
    parser.add_argument("--intf_path", type=str, default="../ExperimentalDNN/data/Measured Data/QPSK/")
    parser.add_argument("--loss_func", type=str, default="MSE", help="amp for amplitude loss func or MSE for MSE loss function.")
    parser.add_argument("--save_suffix", type=str, default = "_1")
    parser.add_argument("--model_dir", type=str, default="Deployment")
    parser.add_argument("--data_len", type=int)
    parser.add_argument("--per_folder", type=int, default=200)
    parser.add_argument("--offset", type=float, default=2e5, help="CW freq offset (Hz)")
    return parser.parse_args()

def freq_shift(x, fs, f0):
    if f0 == 0: return x
    n = np.arange(len(x), dtype=np.float32)
    return x * np.exp(-1j * 2 * np.pi * f0 * n / fs)

class SeparationDataset(Dataset):

    all_mixtures = []  # Class-level storage for all mixtures

    def __init__(self, mixture_dir, groundtruth_dir, interference_dir, total_limit=None, per_folder_limit=10):
        super().__init__()

        #self.mixture_files = sorted(self._gather_all_csvs(mixture_dir))
        #self.groundtruth_files  = sorted(self._gather_all_csvs(groundtruth_dir))
        #self.interference_files = sorted(self._gather_all_csvs(interference_dir))

        self.mixture_files = sorted(self._gather_evenly_sampled_csvs(mixture_dir,total_limit=total_limit, per_folder_limit=per_folder_limit))
        self.groundtruth_files = sorted(self._gather_evenly_sampled_csvs(groundtruth_dir, total_limit=total_limit,per_folder_limit=per_folder_limit))
        self.interference_files = sorted(self._gather_evenly_sampled_csvs(interference_dir, total_limit=total_limit,per_folder_limit=per_folder_limit))

        # Store the metadata here
        self.data_list = []

        # Iterate through the files to read the data and store metadata
        for i, (mix_path, gt_path) in enumerate(zip(self.mixture_files, self.groundtruth_files)):

            int_path = self.interference_files[i % len(self.interference_files)]

            df_mix = pd.read_csv(mix_path,header=0)
            df_gt  = pd.read_csv(gt_path,header=0)
            df_int = pd.read_csv(int_path,header=0)

            if any(df.shape[1] != 2 for df in [df_mix, df_gt, df_int]):
                raise ValueError(f"All CSVs must contain exactly 3 columns: [Time, I, Q]\n{mix_path}\n{gt_path}\n{int_path}")

            # Extract and stack I/Q
            mix  = np.stack([df_mix.iloc[:, 0], df_mix.iloc[:, 1]], axis=-1).astype(np.float32)
            src1 = np.stack([df_gt.iloc[:, 0], df_gt.iloc[:, 1]], axis=-1).astype(np.float32)
            src2 = np.stack([df_int.iloc[:, 0], df_int.iloc[:, 1]], axis=-1).astype(np.float32)
            min_len = min(len(mix), len(src1), len(src2))

            self.data_list.append({
                'mixture':  mix[:min_len],
                'src1':     src1[:min_len],
                'src2':     src2[:min_len],
                'metadata': os.path.basename(mix_path)
            })

            SeparationDataset.all_mixtures = [sample['mixture'] for sample in self.data_list]
    def _gather_all_csvs(self, directory):
        csv_paths = []
        #print(f"Scanning directory: {directory}")  # Debug
        for folder in os.listdir(directory):
            folder_path = os.path.join(directory, folder)
            if os.path.isdir(folder_path):
                #print(f"  Found subfolder: {folder}")  # Debug
                for file in os.listdir(folder_path):
                    if file.endswith(".csv"):
                        csv_paths.append(os.path.join(folder_path, file))
        return sorted(csv_paths)
    
    def _gather_evenly_sampled_csvs(self, directory, total_limit=None, per_folder_limit=10):
        all_csvs = []
        for folder in sorted(os.listdir(directory)):
            folder_path = os.path.join(directory, folder)
            if os.path.isdir(folder_path):
                files = sorted([
                    os.path.join(folder_path, f)
                    for f in os.listdir(folder_path) if f.endswith('.csv')
                ])
                all_csvs.append(files)

        # Flatten and evenly sample across folders
        evenly_sampled = []
        if total_limit is not None:
            num_folders = len(all_csvs)
            per_folder = max(1, total_limit // num_folders)
            
            for files in all_csvs:
                sampled = np.linspace(0, len(files) - 1, per_folder, dtype=int)
                evenly_sampled.extend([files[i] for i in sampled])
        else:
            for files in all_csvs:
                if len(files) == 0:
                    continue
                sampled_idxs = np.linspace(0, len(files) - 1, min(per_folder_limit, len(files)), dtype=int)
                evenly_sampled.extend([files[i] for i in sampled_idxs])


        return sorted(evenly_sampled)

    def __len__(self):
        return len(self.mixture_files)

    def __getitem__(self, idx):
        sample   = self.data_list[idx]
        mixture  = sample['mixture']
        src1     = sample['src1']
        src2     = sample['src2']

        # Convert to torch tensors
        mixture_t = torch.from_numpy(mixture)
        src1_t    = torch.from_numpy(src1)
        src2_t    = torch.from_numpy(src2)

        return mixture_t, src1_t, src2_t, idx

 
def process_dataset(dataset: SeparationDataset, 
                   process_func: callable) -> Dataset:

    class ProcessedDataset(Dataset):
        def __init__(self, original_data: List[Dict[str, Any]]):
            self.data_list = []
            
            for sample in original_data:
                mixture_processed, src1_processed, src2_processed = process_func(
                    sample['mixture'].copy(),
                    sample['src1'].copy(),
                    sample['src2'].copy()
                )

                assert mixture_processed.shape == sample['mixture'].shape
                assert src1_processed.shape == sample['src1'].shape
                assert src2_processed.shape == sample['src2'].shape
                
                self.data_list.append({
                    'mixture': mixture_processed.astype(np.float32),
                    'src1': src1_processed.astype(np.float32),
                    'src2': src2_processed.astype(np.float32),
                    'metadata': sample['metadata']
                })
        
        def __len__(self):
            return len(self.data_list)
            
        def __getitem__(self, idx):
            sample = self.data_list[idx]
            return (
                torch.from_numpy(sample['mixture']),
                torch.from_numpy(sample['src1']),
                torch.from_numpy(sample['src2']),
                idx
            )
    return ProcessedDataset(dataset.data_list)
def shift_all_signals(mix, src1, src2, fs=10e6, f0=2e5):
    # Convert [real, imag] -> complex
    mix_c  = mix[:, 0] + 1j * mix[:, 1]
    src1_c = src1[:, 0] + 1j * src1[:, 1]
    src2_c = src2[:, 0] + 1j * src2[:, 1]

    # Apply frequency shift
    mix_shifted  = freq_shift(mix_c, fs, f0)
    src1_shifted = freq_shift(src1_c, fs, f0)
    src2_shifted = freq_shift(src2_c, fs, f0)

    # Convert back to [real, imag]
    mix_out  = np.stack([mix_shifted.real,  mix_shifted.imag],  axis=-1)
    src1_out = np.stack([src1_shifted.real, src1_shifted.imag], axis=-1)
    src2_out = np.stack([src2_shifted.real, src2_shifted.imag], axis=-1)

    return mix_out, src1_out, src2_out



def save_Deployment_model(model, seqLength=1000, channels=2, filepath="Deployment/Deploy.pth"):

    model_cpu = model.to('cpu')          # Move model to CPU before tracing
    shape_cpu = torch.randn(8, seqLength, channels).to('cpu')  # Dummy input on CPU
    multimodel = torch.jit.trace(model_cpu, shape_cpu)
    stripped = torch.jit.optimize_for_inference(multimodel)
    torch.jit.save(stripped, filepath)
    model.to(device)

def mse_loss_db(est, ref, eps=1e-8):
    import torch.nn.functional as F
    mse = F.mse_loss(est, ref, reduction='mean')
    power_ref = torch.mean(ref ** 2)
    nmse = mse / (power_ref + eps)
    mse_db = 10*torch.log10(nmse + eps)
    return mse_db

def sign_aware_peak_loss(est, ref, eps=1e-8, peak_percent=0.05):
    with torch.no_grad():
        k = max(1, int(peak_percent * ref.shape[1]))
        pos_peaks = torch.topk(ref, k=k//2, dim=1).indices
        neg_peaks = torch.topk(-ref, k=k//2, dim=1).indices
        ref_peaks = torch.cat([pos_peaks, neg_peaks], dim=1)


    est_pk = torch.gather(est, 1, ref_peaks)
    ref_pk = torch.gather(ref, 1, ref_peaks)

    sign_match = torch.sign(est_pk) == torch.sign(ref_pk)
    weighted_errors = torch.where(
        sign_match,
        0.5 * (est_pk - ref_pk).abs(),  
        2.0 * (est_pk.abs() + ref_pk.abs())  
    )
    pk_loss = weighted_errors.mean() / (ref.std() + eps)
    
    return pk_loss

def amplitude_aligned_loss(est, ref, eps=1e-8, alpha=0.6, beta=0.15, gamma = 0.03):

    # Flatten if needed
    if est.dim() == 3:
        B, T, F = est.shape
        est_flat = est.reshape(B, -1)
        ref_flat = ref.reshape(B, -1)
    else:
        est_flat = est
        ref_flat = ref
    
    est_zm = est_flat - est_flat.mean(dim=1, keepdim=True)
    ref_zm = ref_flat - ref_flat.mean(dim=1, keepdim=True)

    dot = torch.sum(est_zm * ref_zm, dim=1, keepdim=True)
    norm_ref = torch.sum(ref_zm ** 2, dim=1, keepdim=True) + eps
    proj = dot / norm_ref * ref_zm

    e = est_zm - proj
    si_snr = 10 * torch.log10(torch.sum(proj**2, dim=1) / (torch.sum(e**2, dim=1) + eps))
    si_snr_loss = -si_snr.mean() 

    peak_loss = sign_aware_peak_loss(est, ref)

    # Amplitude loss (log domain)
    est_rms = torch.sqrt(torch.mean(est_flat**2, dim=1))
    ref_rms = torch.sqrt(torch.mean(ref_flat**2, dim=1))
    amp_ratio = torch.log(est_rms / (ref_rms + eps))
    amp_loss = torch.mean(amp_ratio**2)  # MSE in log domain

    # Attempt to remove DC from the signal
    dc_offset = torch.mean(est_flat, dim=1) 
    dc_penalty = torch.mean(dc_offset**2)

    total_loss = (
        alpha* si_snr_loss + 
        gamma *peak_loss +
        (1-alpha-beta) * amp_loss * 10.0 +  
        beta * dc_penalty * 5.0            
    )
    
    return total_loss


class LossWrapper:
    def __init__(self, loss_name):
        self.loss_name = loss_name
        
    def __call__(self, input, ref):
        if self.loss_name == "MSE":
            return mse_loss_db(input, ref)
        else:  
            return amplitude_aligned_loss(input, ref)
class SELayer1D(nn.Module):
    """Squeeze-and-Excitation for 1D [batch, channels, seq_len]. Very useful for our purpose"""
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        if channels % reduction_ratio != 0:
            raise ValueError('channels is not divisible')
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        squeezed = self.avg_pool(x).view(b, c) 
        excitation = self.fc(squeezed).view(b, c, 1)
        return x * excitation

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, padding=0, **kwargs)
        self.left_padding = kernel_size - 1

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0))
        return super().forward(x)

class CausalLSTM(nn.Module):
    def __init__(
        self,
        input_size=2,   # real+imag (2)
        hidden_size=128,
        num_layers=2,
        use_se_blocks=True,
        bidirectional=False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_se_blocks = use_se_blocks
        self.bidirectional = bidirectional

        self.dir_mult = 2 if bidirectional else 1

        self.input_norm = nn.LayerNorm(input_size)

        self.conv_in = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1
        )

        # LSTM stack
        self.lstm_layers = nn.ModuleList()
        self.se_blocks = nn.ModuleList() if use_se_blocks else None
        
        for i in range(num_layers):
            # LSTM layer
            lstm_input_size = hidden_size if i == 0 else hidden_size * self.dir_mult
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=lstm_input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional
                )
            )
            # SE Block if enabled
            if use_se_blocks:
                self.se_blocks.append(SELayer1D(hidden_size * self.dir_mult))

        # Output
        self.conv_out = nn.Sequential(
            nn.Conv1d(self.dir_mult*hidden_size, self.dir_mult*hidden_size, 3, padding=1),
            nn.ReLU(),
        )
        self.layer_norm = nn.LayerNorm(hidden_size*self.dir_mult)

        self.mask_generator = nn.Sequential(
            nn.Linear(hidden_size*self.dir_mult, hidden_size*self.dir_mult),
            nn.Sigmoid()                          
        )


        self.decoder = nn.Sequential(
            CausalConv1d(self.dir_mult*hidden_size, hidden_size, 3),
            nn.ReLU(),
            CausalConv1d(hidden_size, hidden_size, 3),
            nn.ReLU(),
        )
        self.output_linear = nn.Linear(hidden_size, input_size)
        # Learnable output scaling
        self.output_scale = nn.Parameter(torch.ones(1))
        self.output_bias = nn.Parameter(torch.zeros(1))


    def forward(self, x):

        # Initial convolution
        x = x.permute(0, 2, 1)
        x = self.conv_in(x)
        x = x.permute(0, 2, 1)
        # LSTM processing
        for i in range(self.num_layers):
            # LSTM layer
            lstm_out, _ = self.lstm_layers[i](x)
            

            if self.use_se_blocks:
                lstm_out = lstm_out.permute(0, 2, 1)
                lstm_out = self.se_blocks[i](lstm_out)
                lstm_out = lstm_out.permute(0, 2, 1)
            
            x = lstm_out

        x = self.conv_out(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.layer_norm(x)

        mask = self.mask_generator(x)
        x = x * mask

        # Decoding
        decoded = self.decoder(x.permute(0, 2, 1))
        decoded = decoded.permute(0, 2, 1)
        decoded = self.output_linear(decoded)

        return decoded * self.output_scale+ self.output_bias 

if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = LossWrapper(args.loss_func)
    train = True
    # Build full dataset
    dataset = SeparationDataset(args.mix_path, args.sig_path, args.intf_path, total_limit=args.data_len, per_folder_limit=args.per_folder)
    dataset = process_dataset(dataset,shift_all_signals)

    print("Data list length:", len(dataset.data_list))


    total_len = len(dataset)
    train_len = int(0.75 * total_len)
    val_len   = int(0.10 * total_len)
    test_len  = total_len - train_len - val_len
    power_hyper = 0.2

    train_subset, val_subset, test_subset = random_split(
        dataset,
        lengths=[train_len, val_len, test_len],
        generator=torch.Generator()
    )

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_subset,   batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_subset,  batch_size=args.batch_size, shuffle=True)




    # Initialize single-source separation model
    model = CausalLSTM(
        input_size=2,   # real+imag
        hidden_size=128,
        num_layers=2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = GradScaler(enabled=device=="cuda")

    best_val_loss = float('inf')  # Track best validation loss
    val_losses = []
    train_losses = []
    epoch_bar = tqdm(range(args.epochs), desc="Epochs", file=sys.stderr)
    # --------------------
    # Training Loop
    # --------------------
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0

        for mixture, src1, src2, indices in train_loader:
            mixture = mixture.to(device)
            src1    = src1.to(device)
            scale = torch.sqrt(torch.mean(mixture**2, dim=(1, 2), keepdim=True))
            scale = scale.view(-1, 1, 1)
            mixture /= scale
            src1 /= scale

            optimizer.zero_grad()
            with autocast(device_type=device,enabled=(device == "cuda")):
                outputs = model(mixture)
                loss = loss_fn(outputs, src1)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer); scaler.update()

            running_loss += loss.item()
        sched.step()

        train_epoch_loss = running_loss / len(train_loader)
        train_losses.append(train_epoch_loss)

        # --------------------
        # Validation
        # --------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mixture, src1, src2, indices in val_loader:
                mixture = mixture.to(device)
                src1    = src1.to(device)
                scale = torch.sqrt(torch.mean(mixture**2, dim=(1, 2), keepdim=True))
                scale = scale.view(-1, 1, 1)
                mixture /= scale
                src1 /= scale
                optimizer.zero_grad()
                est = model(mixture)
                loss = loss_fn(est,src1)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # we save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_model{args.save_suffix}.pth")
            save_Deployment_model(model, filepath=f"DeploymentModel{args.save_suffix}.pth")
    model.load_state_dict(torch.load(f"best_model{args.save_suffix}.pth"))
    model.eval()
    test_running_loss = 0.0
    param_count = sum(p.numel() for p in model.parameters())
    print("Total parameters:", param_count)

    with torch.no_grad():
        for mixture, src1, src2, indices in test_loader:
            mixture = mixture.to(device)
            src1    = src1.to(device)
            scale = torch.sqrt(torch.mean(mixture**2, dim=(1, 2), keepdim=True))
            scale = scale.view(-1, 1, 1)
            mixture /= scale
            src1 /= scale
            if(args.scale=="on"):
                est, scale_in = model(mixture)
                est = est*scale_in
            else:
                est = model(mixture)
            test_loss = loss_fn(est, src1)
            test_running_loss += test_loss.item()

    test_loss = test_running_loss / len(test_loader)
    print(test_loss)

