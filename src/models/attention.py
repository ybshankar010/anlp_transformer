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
    def __init__(self,d_model, num_heads,dropout = 0.1, use_rope=False) -> None:
        super().__init__()

        if d_model % num_heads != 0 :
            raise ValueError("Invalid params : d_model should be divisible by num heads")

        self.head_dim = d_model // num_heads

        self.num_heads = num_heads
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def apply_rope(self, x):
        batch_size, num_heads, seq_len, head_dim = x.shape

        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")

        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, head_dim, 2, device=x.device, dtype=x.dtype)
            * (-math.log(10000.0) / head_dim)
        )
        angles = positions * frequencies
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated = torch.empty_like(x)
        rotated[..., 0::2] = x_even * cos - x_odd * sin
        rotated[..., 1::2] = x_even * sin + x_odd * cos
        return rotated

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

        if self.use_rope:
            q = self.apply_rope(q)
            k = self.apply_rope(k)

        attn_outputs, attn_weights = self.attention(q,k,v,mask)

        attn_outputs = self.combine_heads(attn_outputs)

        output = self.out_proj(attn_outputs)

        return output, attn_weights 


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads=None, dropout=0.1, use_rope=False) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("Invalid params: d_model should be divisible by num_heads")

        if num_kv_heads is None:
            num_kv_heads = max(1, num_heads // 2)

        if num_heads % num_kv_heads != 0:
            raise ValueError("Invalid params: num_heads should be divisible by num_kv_heads")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.kv_repeat = num_heads // num_kv_heads
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def split_query_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def split_kv_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        return x.transpose(1, 2)

    def repeat_kv_heads(self, x):
        return x.repeat_interleave(self.kv_repeat, dim=1)

    def combine_heads(self, x):
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * head_dim)

    def apply_rope(self, x):
        batch_size, num_heads, seq_len, head_dim = x.shape

        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")

        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, head_dim, 2, device=x.device, dtype=x.dtype)
            * (-math.log(10000.0) / head_dim)
        )
        angles = positions * frequencies
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated = torch.empty_like(x)
        rotated[..., 0::2] = x_even * cos - x_odd * sin
        rotated[..., 1::2] = x_even * sin + x_odd * cos
        return rotated

    def forward(self, q, k, v, mask=None):
        q = self.split_query_heads(self.q_proj(q))
        k = self.split_kv_heads(self.k_proj(k))
        v = self.split_kv_heads(self.v_proj(v))

        if self.use_rope:
            q = self.apply_rope(q)
            k = self.apply_rope(k)

        k = self.repeat_kv_heads(k)
        v = self.repeat_kv_heads(v)

        attn_outputs, attn_weights = self.attention(q, k, v, mask)
        attn_outputs = self.combine_heads(attn_outputs)
        output = self.out_proj(attn_outputs)

        return output, attn_weights


def build_attention(attention_type, d_model, num_heads, dropout=0.1, use_rope=False):
    if attention_type == "mha":
        return MultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
        )

    if attention_type == "gqa":
        return GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
        )

    raise ValueError(f"Unsupported attention_type: {attention_type}")

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
    
