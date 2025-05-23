


import torch.nn as nn
import torch
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


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, padding=0, **kwargs)
        self.left_padding = kernel_size - 1

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0))
        return super().forward(x)

class LSTMSingleSource(nn.Module):
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

        # Input processing
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
            #self.PermuteLayerNorm(self.dir_mult*hidden_size)
        )
        self.layer_norm = nn.LayerNorm(hidden_size*self.dir_mult)

        # Mask generator - operates on time dimension
        self.mask_generator = nn.Sequential(
            nn.Linear(hidden_size*self.dir_mult, hidden_size*self.dir_mult),
            nn.Sigmoid()                          
        )


        self.decoder = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),  # Pool time dimension → (B, C, 1)
                nn.Flatten(),             # → (B, C)
                nn.Linear(hidden_size * self.dir_mult, input_size)  # → (B, 2) for real+imag
            )
        #self.decoder = nn.Conv1d(hidden_size*self.dir_mult, input_size, kernel_size=3, padding=1)
        
        # Learnable output scaling
        self.output_scale = nn.Parameter(torch.ones(1))
        self.output_bias = nn.Parameter(torch.zeros(1))

    class PermuteLayerNorm(nn.Module):
        def __init__(self, normalized_shape):
            super().__init__()
            self.norm = nn.LayerNorm(normalized_shape)
        
        def forward(self, x):
            x = x.permute(0, 2, 1)  
            x = self.norm(x)
            return x.permute(0, 2, 1) 

    def forward(self, x):
        B, T, _ = x.shape

        x_normalized = x #/ scale

        x = x_normalized
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
        output = self.decoder(x.permute(0, 2, 1))
        return {"gain": output}