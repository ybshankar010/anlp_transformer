import math
import torch
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000, dropout = 0.1) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model)
        )

        pe = torch.zeros(max_len,d_model)
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe",pe)


    def forward(self, x):
        seq_len = x.size(1)

        x = x + self.pe[:,:seq_len,:]

        return self.dropout(x)


def test_positional_encoding():
    positional_encoding = SinusoidalPositionalEncoding(
        d_model = 256,
        max_len = 2000,
        dropout = 0.1 
    )

    x = torch.randn(16,1317, 256)

    y = positional_encoding(x)

    logger.debug("shape of y :: %s",y.shape)
