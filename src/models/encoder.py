import torch
import torch.nn as nn
import logging
from .attention import build_attention
from .feedforward import PositionwiseFeedforward
from .norm import build_norm


logger = logging.getLogger(__name__)

class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        ffn_dim,
        dropout=0.1,
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
        self.feedforward = PositionwiseFeedforward(d_model=d_model,ffn_dim=ffn_dim,dropout=dropout)

        self.norm1 = build_norm(norm_type, d_model)
        self.norm2 = build_norm(norm_type, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x, src_mask = None):
        norm_x = self.norm1(x)
        output,attn_weights = self.attention(norm_x,norm_x,norm_x,src_mask)
        x = x + self.dropout1(output)

        ffn_output = self.feedforward(self.norm2(x))
        x = x + self.dropout2(ffn_output)
        
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        ffn_dim,
        num_layers,
        dropout=0.1,
        attention_type="mha",
        norm_type="layernorm",
        use_rope=False,
    ) -> None:
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attention_type=attention_type,
                norm_type=norm_type,
                use_rope=use_rope,
            )
            for _ in range(num_layers)
        ])

    def forward(self,x,src_mask=None):

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x,src_mask)

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
