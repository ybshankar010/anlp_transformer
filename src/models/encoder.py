from typing import Any

import torch
import torch.nn as nn
import logging
from .attention import MultiheadAttention
from .feedforward import PositionwiseFeedforward


logger = logging.getLogger(__name__)

class EncoderLayer(nn.Module):
    def __init__(self,d_model, num_heads,ffn_dim, dropout=0.1) -> None:
        super().__init__()

        self.attention = MultiheadAttention(d_model=d_model,num_heads=num_heads,dropout=dropout)
        self.feedforward = PositionwiseFeedforward(d_model=d_model,ffn_dim=ffn_dim,dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x, src_mask = None):
        output,attn_weights = self.attention(x,x,x,src_mask)
        x = self.norm1(x + self.dropout1(output))

        ffn_output = self.feedforward(x)
        x = self.norm2(x +self.dropout2(ffn_output))
        
        return x


class Encoder(nn.Module):
    def __init__(self,d_model,num_heads,ffn_dim, num_layers,dropout=0.1) -> None:
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model=d_model,num_heads=num_heads,ffn_dim=ffn_dim,dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self,x,src_mask=None):

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        return x


def test_encoder_layer():
    x = torch.randn(16,100,256)
    layer = EncoderLayer(d_model=256,num_heads=8,ffn_dim=1024)

    y = layer(x)
    logger.debug("=========== TEST Encoder Layer ==============")
    logger.debug("Shape of y after encoder layer %s",y.shape)

    encoder = Encoder(d_model=256,num_heads=8,ffn_dim=1024,num_layers=8)
    y = encoder(x)

    logger.debug("Shape of y for passing through encoder stack %s",y.shape)