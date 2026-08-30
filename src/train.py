import logging
import os
import json
from dataclasses import asdict
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
import wandb
from huggingface_hub import HfApi
from dotenv import load_dotenv
from src.plots import plot_training_history
load_dotenv()

logger = logging.getLogger(__name__)

def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()

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

    return total_loss / len(train_loader)

def evaluate_loss(model, data_loader, loss_fn, device):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            src_ids = batch["cipher_text"].to(device)
            decoder_input_ids = batch["plain_text_input_ids"].to(device)
            target_ids = batch["plain_text_target_ids"].to(device)

            src_padding_mask = batch["cipher_padding_mask"].to(device)
            target_padding_mask = batch["plain_text_padding_mask"].to(device)

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

            total_loss += loss.item()

    return total_loss / len(data_loader)

def save_checkpoint(
    checkpoint_dir,
    model,
    optimizer,
    config,
    cipher_tokenizer,
    plain_tokenizer,
    epoch,
    train_loss,
    validation_loss,
    history=None,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        },
        os.path.join(checkpoint_dir, "model.pt"),
    )

    cipher_tokenizer.save(os.path.join(checkpoint_dir, "cipher_tokenizer.json"))
    plain_tokenizer.save(os.path.join(checkpoint_dir, "plain_tokenizer.json"))

    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)

    with open(os.path.join(checkpoint_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "history": history or [],
            },
            f,
            indent=2,
        )

    with open(os.path.join(checkpoint_dir, "training_history.json"), "w") as f:
        json.dump(history or [], f, indent=2)

def upload_checkpoint_to_huggingface(checkpoint_dir, repo_id, path_in_repo):
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=True,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=path_in_repo,
    )

def train_c1():
    config = EXPERIMENTS[0]

    run = wandb.init(
        project=config.wandb_project,
        name=config.name,
        config=asdict(config),
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    dataset = CipherPlainDataset()
    train_dataset, validation_dataset, _ = create_datasplits(dataset, config)

    cipher_texts = [item["cipher_text"] for item in train_dataset]
    plain_texts = [item["plain_text"] for item in train_dataset]

    cipher_tokenizer = BPETokenizer(vocab_size=100)
    plain_tokenizer = BPETokenizer(vocab_size=100)

    cipher_tokenizer.train(cipher_texts[:1000])
    plain_tokenizer.train(plain_texts[:1000])

    collator = CipherPlainDatasetCollator(
        cipher_tokenizer=cipher_tokenizer,
        plain_text_tokenizer=plain_tokenizer,
        max_target_len=config.max_target_len,
        max_src_len=config.max_src_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

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
        max_src_len=config.max_src_len,
        max_target_len=config.max_target_len,
        dropout=config.dropout,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=plain_tokenizer.pad_token_id)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    checkpoint_dir = config.checkpoint_dir

    best_validation_loss = float("inf")
    history = []

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        validation_loss = evaluate_loss(
            model=model,
            data_loader=validation_loader,
            loss_fn=loss_fn,
            device=device,
        )

        logger.info(
            "Epoch %s | train loss %.4f | validation loss %.4f",
            epoch + 1,
            train_loss,
            validation_loss,
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                optimizer=optimizer,
                config=config,
                cipher_tokenizer=cipher_tokenizer,
                plain_tokenizer=plain_tokenizer,
                epoch=epoch + 1,
                train_loss=train_loss,
                validation_loss=validation_loss,
                history=history,
            )
            logger.info("Saved best checkpoint to %s", checkpoint_dir)

    plot_training_history(
        history=history,
        output_path=os.path.join(checkpoint_dir, "training_loss.png"),
    )

    artifact = wandb.Artifact(
        name="c1-transformer",
        type="model",
        metadata={
            "repo_id": config.hf_repo_id,
            "path_in_repo": config.hf_path_in_repo,
            "best_validation_loss": best_validation_loss,
        },
    )
    artifact.add_dir(checkpoint_dir)
    run.log_artifact(artifact)

    if config.upload_to_hf:
        upload_checkpoint_to_huggingface(
            checkpoint_dir=checkpoint_dir,
            repo_id=config.hf_repo_id,
            path_in_repo=config.hf_path_in_repo,
        )
    wandb.finish()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    train_c1()
