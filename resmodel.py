import torch
import torch.nn as nn
import torch.nn.functional as F





########################################################################
# small residual block
########################################################################
class ResBlock1D(nn.Module):
    def __init__(self, ch, dilation=1):
        super().__init__()
        pad = dilation
        self.c1 = nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation)
        self.c2 = nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation)
        self.g1 = nn.GroupNorm(8, ch)
        self.g2 = nn.GroupNorm(8, ch)
    def forward(self,x):
        y = F.silu(self.g1(self.c1(x)))
        y = self.g2(self.c2(y))
        return F.silu(x + y)

########################################################################
# Hybrid beacon estimator  (wide + narrow)
########################################################################
class HybridBeaconEstimator(nn.Module):
    def __init__(self, in_ch=2, base=64, lstm_h=64):
        super().__init__()
        # wide encoder
        self.proj = nn.Conv1d(in_ch, base, 3, padding=1)
        self.res  = nn.ModuleList([ResBlock1D(base, d) for d in (1,2,4,8)])

        # narrow encoder
        self.n_proj = nn.Conv1d(in_ch, base//2, 3, padding=1)
        self.n_res  = nn.ModuleList([ResBlock1D(base//2,1),
                                     ResBlock1D(base//2,2)])

        # LSTM on concatenated features
        self.lstm = nn.LSTM(base+base//2, lstm_h, 1,
                            batch_first=True, bidirectional=False)

        # heads
        self.head_g  = nn.Linear(lstm_h, 2)

    def forward(self, x):            # x (B,2,N)  real I/Q
    # wide path
        w = self.proj(x)
        for blk in self.res:
            w = blk(w)

    # narrow path
        nb = self.n_proj(x)
        for blk in self.n_res:
            nb = blk(nb)
        nb = F.interpolate(nb, size=w.shape[-1],
                       mode='linear', align_corners=False)

    # concat & LSTM
        feat = torch.cat([w, nb], dim=1).transpose(1, 2)   # (B,N,C’)
        out, _ = self.lstm(feat)
        v = out.mean(1)
        return {"gain": self.head_g(v)}                    # (B,2)

############################################################
# Example usage
############################################################
if __name__ == "__main__":
    B, N = 4, 1000
    dummy = torch.randn(B, N, 2)
    model = HybridBeaconEstimator()
    out = model(dummy)
    print({k: v.shape for k, v in out.items()})