import torch
import torch.nn as nn


class lstm(nn.LSTM):
    def __init__(self,config):
        super().__init__(
            input_size = config.d_model,
            hidden_size = config.d_model,
            num_layers = config.depth,
            batch_first  = True,
            dropout = config.dropout
        )
    def forward(self,x,mask):
        bz,n_fm,dim = x.shape
        last_visible = torch.arange(n_fm).expand(bz,n_fm).to(x.device) * mask
        last_visible = last_visible.max(-1)[0].long()
        
    
        x = torch.where(
            (mask == 1).unsqueeze(-1).expand_as(x),
            x,
            x[torch.arange(bz).to(x.device).long(),last_visible].unsqueeze(1).expand_as(x)
        )
        return super().forward(x)[0]