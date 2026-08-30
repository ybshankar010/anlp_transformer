import math
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


class MultiheadAttention(nn.Module):
    def __init__(self,d_model, num_heads,dropout = 0.1) -> None:
        super().__init__()

        if d_model % num_heads != 0 :
            raise ValueError("Invalid params : d_model should be divisible by num heads")

        self.head_dim = d_model // num_heads

        self.num_heads = num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size,seq_len, self.num_heads,self.head_dim)

        return x.transpose(1,2)

    def combine_heads(self, x):
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(1,2).contiguous()
        return x.view(batch_size, seq_len, num_heads * head_dim)

    def forward(self, q,k,v,mask=None):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        attn_outputs, attn_weights = self.attention(q,k,v,mask)

        attn_outputs = self.combine_heads(attn_outputs)

        output = self.out_proj(attn_outputs)

        return output, attn_weights 

def test_scaled_dotproduct():
    q = torch.randn(2,4,5,8)
    k = torch.randn(2,4,6,8)
    v = torch.randn(2,4,6,8)

    scaled_attn = ScaledDotProductAttention(0.1)

    outputs, weights = scaled_attn(q,k,v)

    logger.debug("Output shape :: %s",outputs.shape)
    logger.debug("Weights shape :: %s",weights.shape)

    logger.debug("Testing multihead attention")
    mha = MultiheadAttention(d_model=256, num_heads=8, dropout=0.1)

    x = torch.randn(16, 100, 256)
    outputs, weights = mha(x, x, x)

    logger.debug("Outputs shape :: %s", outputs.shape)
    logger.debug("Weights shape :: %s", weights.shape)

    return
    