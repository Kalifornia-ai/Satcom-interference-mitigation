


import torch.nn as nn
import torch.nn.functional as F






class SELayer1D(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Sequential(
            nn.Linear(channels, channels // r), nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels), nn.Sigmoid())
    def forward(self, x):
        b,c,_ = x.size()
        y = self.avg(x).view(b,c)
        y = self.fc(y).view(b,c,1)
        return x * y

# lstm_model.py  (or resmodel.py – wherever the class lives)
class LSTMSeperatorSingle(nn.Module):
    def __init__(self, input_size=2, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.conv_in = nn.Conv1d(input_size, hidden, 3, padding=1)

        self.lstm   = nn.ModuleList()
        self.se     = nn.ModuleList()
        for i in range(layers):
            in_dim = hidden if i == 0 else 2 * hidden
            self.lstm.append(nn.LSTM(in_dim, hidden,
                                     num_layers=1, batch_first=True,
                                     bidirectional=True))
            self.se.append(SELayer1D(2 * hidden))

        self.conv_out = nn.Conv1d(2 * hidden, 2 * hidden, 3, padding=1)
        self.norm     = nn.LayerNorm(2 * hidden)

        # NEW – 2-unit head that produces real & imag parts of g
        self.head_g   = nn.Linear(2 * hidden, input_size)

    # ------------------------------------------------------------------
    def forward(self, x):                  # x: (B, N, 2)
        x = x.transpose(1, 2)              # (B, 2, N)
        x = self.conv_in(x)                # (B, H, N)
        x = x.transpose(1, 2)              # (B, N, H)

        for lstm, se in zip(self.lstm, self.se):
            x, _ = lstm(x)                 # (B, N, 2H)
            x    = se(x.transpose(1, 2)).transpose(1, 2)

        x = self.conv_out(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x)                   # (B, N, 2H)

        v = x.mean(1)                      # temporal mean → (B, 2H)
        return {"gain": self.head_g(v)}    # (B, 2)
