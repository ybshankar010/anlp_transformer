import math
from typing import Any
import torch
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)


    def forward(self,q,k,v,mask=None):
        scores = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(q.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask,float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)

        return output, attn_weights



def test_scaled_dotproduct():
    q = torch.randn(2,4,5,8)
    k = torch.randn(2,4,6,8)
    v = torch.randn(2,4,6,8)

    scaled_attn = ScaledDotProductAttention(0.1)

    outputs, weights = scaled_attn(q,k,v)

    logger.debug("Output shape :: %s",outputs.shape)
    logger.debug("Weights shape :: %s",weights.shape)

    return
    