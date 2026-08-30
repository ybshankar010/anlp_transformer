import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import (
    CipherPlainDataset,
    CipherPlainDatasetCollator,
    create_datasplits,
)
from src.tokenizer import BPETokenizer
from src.models.transformer import Seq2SeqTransformer
from src.utils import EXPERIMENTS

logger = logging.getLogger(__name__)

def train_tiny_overfit():
    config = EXPERIMENTS[0]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    dataset = CipherPlainDataset()
    train_dataset, _, _ = create_datasplits(dataset, config)

    overfit_size = 32
    small_train_dataset = torch.utils.data.Subset(
        train_dataset,
        range(overfit_size),
    )

    cipher_texts = [item["cipher_text"] for item in small_train_dataset]
    plain_texts = [item["plain_text"] for item in small_train_dataset]

    cipher_tokenizer = BPETokenizer(vocab_size=100)
    plain_tokenizer = BPETokenizer(vocab_size=100)

    cipher_tokenizer.train(cipher_texts)
    plain_tokenizer.train(plain_texts)

    collator = CipherPlainDatasetCollator(
        cipher_tokenizer=cipher_tokenizer,
        plain_text_tokenizer=plain_tokenizer,
        max_target_len=config.max_target_len,
    )

    train_loader = DataLoader(
        small_train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collator,
    )

    max_src_len = max(len(cipher_tokenizer.encode(text)) for text in cipher_texts)

    model = Seq2SeqTransformer(
        src_vocab_size=cipher_tokenizer.get_vocab_size(),
        target_vocab_size=plain_tokenizer.get_vocab_size(),
        src_pad_id=cipher_tokenizer.pad_token_id,
        target_pad_id=plain_tokenizer.pad_token_id,
        d_model=config.d_model,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        max_src_len=max_src_len,
        max_target_len=config.max_target_len,
        dropout=config.dropout,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=plain_tokenizer.pad_token_id
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    model.train()

    for epoch in range(10):
        total_loss = 0.0

        for batch in train_loader:
            src_ids = batch["cipher_text"].to(device)
            decoder_input_ids = batch["plain_text_input_ids"].to(device)
            target_ids = batch["plain_text_target_ids"].to(device)
            src_padding_mask = batch["cipher_padding_mask"].to(device)
            target_padding_mask = batch["plain_text_padding_mask"].to(device)

            optimizer.zero_grad()

            logits = model(
                src_ids=src_ids,
                decoder_input_ids=decoder_input_ids,
                src_padding_mask=src_padding_mask,
                target_padding_mask=target_padding_mask,
            )

            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info("Epoch %s | loss %.4f", epoch + 1, avg_loss)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    train_tiny_overfit()