import torch
import torch.nn as nn

from src.constants import (
    BIT_PAD_ID,
    BIT_VOCAB_SIZE,
    BYTE_PAD_ID,
    BYTE_VOCAB_SIZE,
)
from src.models.decoder import Decoder
from src.models.encoder import Encoder
from src.models.embedding import TokenEmbedding
from src.models.masks import create_causal_mask
from src.models.positional import SinusoidalPositionalEncoding


class LocalByteEncoder(nn.Module):
    def __init__(self, local_vocab_size, local_dim, d_model, patch_size, pad_id, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.pad_id = pad_id
        self.embedding = nn.Embedding(
            local_vocab_size,
            local_dim,
            padding_idx=pad_id,
        )
        self.projection = nn.Linear(patch_size * local_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def pad_to_patch_multiple(self, ids, padding_mask):
        batch_size, seq_len = ids.shape
        remainder = seq_len % self.patch_size

        if remainder == 0:
            return ids, padding_mask

        pad_len = self.patch_size - remainder
        id_padding = torch.full(
            (batch_size, pad_len),
            self.pad_id,
            dtype=ids.dtype,
            device=ids.device,
        )
        mask_padding = torch.ones(
            batch_size,
            pad_len,
            dtype=torch.bool,
            device=ids.device,
        )

        return (
            torch.cat([ids, id_padding], dim=1),
            torch.cat([padding_mask, mask_padding], dim=1),
        )

    def forward(self, ids, padding_mask):
        ids, padding_mask = self.pad_to_patch_multiple(ids, padding_mask)
        batch_size, seq_len = ids.shape
        num_patches = seq_len // self.patch_size

        embedded = self.embedding(ids)
        patches = embedded.view(batch_size, num_patches, self.patch_size * embedded.size(-1))
        patch_embeddings = self.projection(patches)
        patch_embeddings = self.dropout(patch_embeddings)

        patch_padding_mask = padding_mask.view(
            batch_size,
            num_patches,
            self.patch_size,
        ).all(dim=-1)

        return patch_embeddings, patch_padding_mask


class LocalByteDecoder(nn.Module):
    def __init__(self, d_model, byte_vocab_size):
        super().__init__()
        self.projection = nn.Linear(d_model, byte_vocab_size)

    def forward(self, x):
        return self.projection(x)


class BLTSeq2SeqTransformer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        ffn_dim,
        encoder_layers,
        decoder_layers,
        max_src_len,
        max_target_len,
        patch_size,
        local_dim,
        dropout=0.1,
    ):
        super().__init__()
        max_patches = (max_src_len + patch_size - 1) // patch_size

        self.local_encoder = LocalByteEncoder(
            local_vocab_size=BIT_VOCAB_SIZE,
            local_dim=local_dim,
            d_model=d_model,
            patch_size=patch_size,
            pad_id=BIT_PAD_ID,
            dropout=dropout,
        )
        self.target_embedding = TokenEmbedding(
            vocab_size=BYTE_VOCAB_SIZE,
            d_model=d_model,
            pad_id=BYTE_PAD_ID,
        )
        self.src_positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_patches,
            dropout=dropout,
        )
        self.target_positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_target_len,
            dropout=dropout,
        )
        self.encoder = Encoder(
            d_model=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=encoder_layers,
            dropout=dropout,
            attention_type="mha",
            norm_type="layernorm",
            use_rope=False,
        )
        self.decoder = Decoder(
            d_model=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=decoder_layers,
            dropout=dropout,
            attention_type="mha",
            norm_type="layernorm",
            use_rope=False,
        )
        self.local_decoder = LocalByteDecoder(
            d_model=d_model,
            byte_vocab_size=BYTE_VOCAB_SIZE,
        )

    def forward(self, src_ids, decoder_input_ids, src_padding_mask=None, target_padding_mask=None):
        if src_padding_mask is None:
            src_padding_mask = src_ids == BIT_PAD_ID

        src, patch_padding_mask = self.local_encoder(src_ids, src_padding_mask)
        src = self.src_positional_encoding(src)
        src_mask = patch_padding_mask.unsqueeze(1).unsqueeze(2)
        encoder_output = self.encoder(src, src_mask)

        target_mask = None
        if target_padding_mask is not None:
            target_mask = target_padding_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = create_causal_mask(
                decoder_input_ids.size(1),
                device=decoder_input_ids.device,
            )
            target_mask = target_mask | causal_mask

        tgt = self.target_embedding(decoder_input_ids)
        tgt = self.target_positional_encoding(tgt)
        decoder_output = self.decoder(tgt, encoder_output, target_mask, src_mask)
        return self.local_decoder(decoder_output)
