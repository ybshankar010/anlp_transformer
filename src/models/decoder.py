from typing import Any

from .attention import build_attention
from .feedforward import PositionwiseFeedforward
from .norm import build_norm

import torch
import torch.nn as nn
import logging


logger = logging.getLogger(__name__)


class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        ffn_dim,
        dropout = 0.1,
        attention_type="mha",
        norm_type="layernorm",
        use_rope=False,
    ) -> None:
        super().__init__()

        self.attention = build_attention(
            attention_type=attention_type,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
        )
        self.cross_attention = build_attention(
            attention_type=attention_type,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
        )
        self.feedforward = PositionwiseFeedforward(d_model=d_model,ffn_dim=ffn_dim,dropout=dropout)

        self.norm1 = build_norm(norm_type, d_model)
        self.norm2 = build_norm(norm_type, d_model)
        self.norm3 = build_norm(norm_type, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)


    def forward(self,x,encoder_output,tgt_mask = None,src_mask = None) :
        norm_x = self.norm1(x)
        self_attn_output, _ = self.attention(norm_x,norm_x,norm_x,tgt_mask)
        x = x + self.dropout1(self_attn_output)

        cross_attn_output,_ = self.cross_attention(self.norm2(x),encoder_output,encoder_output,src_mask)
        x = x + self.dropout2(cross_attn_output)

        ffn_output = self.feedforward(self.norm3(x))
        x = x + self.dropout3(ffn_output)

        return x


class Decoder(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        ffn_dim,
        num_layers,
        dropout = 0.1,
        attention_type="mha",
        norm_type="layernorm",
        use_rope=False,
    ) -> None:
        super().__init__()

        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    attention_type=attention_type,
                    norm_type=norm_type,
                    use_rope=use_rope,
                )
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
