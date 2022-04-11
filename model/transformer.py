import torch
import torch.nn as nn
import math

def build_position_embedding(pe_type,n_frame,d_model):
    if pe_type is None or pe_type == 'no':
        return 0
    elif pe_type == 'learnable':
        return nn.Parameter(torch.zeros(n_frame,d_model),requires_grad=True)
    elif pe_type == 'sin-cos':
        embed = nn.Parameter(torch.zeros(n_frame,d_model),requires_grad=False)
        for i in range(d_model):
            t = math.pow(10000,i/d_model)
            if i % 2==0:
                embed[:,i] = torch.sin(torch.arange(n_frame)  / t)
            else:
                embed[:,i] = torch.cos(torch.arange(n_frame) /  t)
        return embed
    else:
        raise NotImplementedError(pe_type)


class transformer(nn.Module):
    def __init__(self,config,length):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model = config.d_model,
                nhead = config.nhead,
                dim_feedforward = config.dff,
                activation = 'gelu',
                dropout = config.dropout
                )
            for _ in range(config.depth)
        ])
        self.pe = build_position_embedding(config.pe_type, length, config.d_model)
        self.ln = nn.LayerNorm(config.d_model)

    def forward(self,x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        x = x + self.pe                     # B T C
        x = x.permute(1,0,2)                # T B C
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x).permute(1,0,2)       # B T C
        return x

