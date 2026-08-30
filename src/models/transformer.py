import torch
import torch.nn as nn
import logging

from .embedding import TokenEmbedding
from .positional import SinusoidalPositionalEncoding
from .encoder import Encoder
from .decoder import Decoder
from .masks import create_causal_mask

logger = logging.getLogger(__name__)

class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 target_vocab_size,
                 src_pad_id,
                 target_pad_id,
                 d_model,
                 num_heads,
                 ffn_dim,
                 encoder_layers,
                 decoder_layers,
                 max_src_len,
                 max_target_len,
                 dropout=0.1,) -> None:
        super().__init__()
        self.src_embedding = TokenEmbedding(src_vocab_size,d_model=d_model,pad_id=src_pad_id)
        self.target_embedding = TokenEmbedding(target_vocab_size,d_model=d_model,pad_id=target_pad_id)

        self.src_positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len= max_src_len,
            dropout=dropout
        )

        self.target_positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_target_len,
            dropout=dropout
        )

        self.encoder = Encoder(d_model=d_model,num_heads=num_heads,ffn_dim=ffn_dim,num_layers=encoder_layers,dropout=dropout)
        self.decoder = Decoder(d_model=d_model,num_heads=num_heads,ffn_dim=ffn_dim,num_layers=decoder_layers,dropout=dropout)

        self.output_projection = nn.Linear(d_model,target_vocab_size)


    def forward(self,src_ids,decoder_input_ids,src_padding_mask = None,target_padding_mask = None):
        src_mask = None
        if src_padding_mask is not None:
            src_mask = src_padding_mask.unsqueeze(1).unsqueeze(2)

        target_mask = None
        if target_padding_mask is not None:
            target_mask = target_padding_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = create_causal_mask(
                decoder_input_ids.size(1),
                device=decoder_input_ids.device
            )

            target_mask = target_mask | causal_mask

        src = self.src_embedding(src_ids)
        src = self.src_positional_encoding(src)
        encoder_output = self.encoder(src,src_mask)

        tgt = self.target_embedding(decoder_input_ids)
        tgt = self.target_positional_encoding(tgt)
        decoder_output = self.decoder(tgt,encoder_output,target_mask,src_mask)

        logits = self.output_projection(decoder_output)
        return logits


def test_seq2seqtransformer():
    transformer = Seq2SeqTransformer(
        src_vocab_size=100,
        target_vocab_size=100,
        src_pad_id=0,
        target_pad_id=0,
        d_model=256,
        num_heads=8,
        ffn_dim=1024,
        encoder_layers=4,
        decoder_layers=4,
        max_src_len=2000,
        max_target_len=256,
        )

    src_ids = torch.randint(0, 100, (16, 300))
    decoder_input_ids = torch.randint(0, 100, (16, 50))

    src_padding_mask = src_ids == 0
    target_padding_mask = decoder_input_ids == 0

    logits = transformer(
        src_ids,
        decoder_input_ids,
        src_padding_mask,
        target_padding_mask,
    )

    logger.debug("=====Test Transformer ====")
    logger.debug("Shape of logits %s",logits.shape)


    


