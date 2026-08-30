from typing import Any

from .attention import MultiheadAttention
from .feedforward import PositionwiseFeedforward

import torch
import torch.nn as nn
import logging


logger = logging.getLogger(__name__)


class DecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, ffn_dim, dropout = 0.1) -> None:
        super().__init__()

        self.attention = MultiheadAttention(d_model=d_model,num_heads=num_heads,dropout=dropout)
        self.cross_attention = MultiheadAttention(d_model=d_model,num_heads=num_heads,dropout=dropout)
        self.feedforward = PositionwiseFeedforward(d_model=d_model,ffn_dim=ffn_dim,dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)


    def forward(self,x,encoder_output,tgt_mask = None,src_mask = None) :
        self_attn_output, _ = self.attention(x,x,x,tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))

        cross_attn_output,_ = self.cross_attention(x,encoder_output,encoder_output,src_mask)
        x = self.norm2(x+self.dropout2(cross_attn_output))

        ffn_output = self.feedforward(x)
        x = self.norm3(x+self.dropout3(ffn_output))

        return x


class Decoder(nn.Module):

    def __init__(self, d_model, num_heads, ffn_dim, num_layers,dropout = 0.1) -> None:
        super().__init__()

        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(d_model=d_model,num_heads=num_heads,ffn_dim=ffn_dim,dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, encoder_output,tgt_mask = None,src_mask = None):

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x,encoder_output,tgt_mask,src_mask)

        return x



def test_decoder_layer():
    x = torch.randn(16,50,256)
    encoder_output = torch.randn(16,100,256)

    decoder_layer = DecoderLayer(d_model=256,num_heads=8,ffn_dim=1024)
    y = decoder_layer(x,encoder_output)

    logger.debug("======== TEST DECODER =========")
    logger.debug("Shape of y %s after 1 layer",y.shape)

    decoder = Decoder(d_model=256,num_heads=8,ffn_dim=1024,num_layers=8)
    y = decoder(x,encoder_output)
    logger.debug("Shape of y %s after 8 layers",y.shape)

