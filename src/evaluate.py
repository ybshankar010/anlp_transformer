import argparse
import csv
import json
import logging
import os
from dataclasses import asdict

import torch

from src.dataset import CipherPlainDataset, create_datasplits
from src.metrics import average_metrics, compute_prediction_metrics
from src.models.transformer import Seq2SeqTransformer
from src.plots import (
    plot_ablation_summary,
    plot_edit_distance,
    plot_metric_summary,
    plot_prediction_lengths,
    plot_training_history,
)
from src.tokenizer import BPETokenizer
from src.utils import EXPERIMENTS, ExperimentConfig


logger = logging.getLogger(__name__)

METRIC_FIELDNAMES = [
    "bit_level_accuracy",
    "sequence_accuracy",
    "levenshtein_distance",
    "bleu",
    "rouge_1",
    "rouge_2",
    "rouge_l",
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_checkpoint(checkpoint_dir, device):
    config = ExperimentConfig(**load_json(os.path.join(checkpoint_dir, "config.json")))
    cipher_tokenizer = BPETokenizer.load(os.path.join(checkpoint_dir, "cipher_tokenizer.json"))
    plain_tokenizer = BPETokenizer.load(os.path.join(checkpoint_dir, "plain_tokenizer.json"))

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
        positional_encoding=config.positional_encoding,
        attention_type=config.attention_type,
        norm_type=config.norm_type,
        dropout=config.dropout,
    ).to(device)

    checkpoint = torch.load(
        os.path.join(checkpoint_dir, "model.pt"),
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config, cipher_tokenizer, plain_tokenizer


def greedy_decode(model, cipher_text, cipher_tokenizer, plain_tokenizer, config, device):
    src_ids = torch.tensor(
        [cipher_tokenizer.encode(cipher_text)[: config.max_src_len]],
        dtype=torch.long,
        device=device,
    )
    src_padding_mask = src_ids == cipher_tokenizer.pad_token_id

    generated_ids = [plain_tokenizer.bos_token_id]

    with torch.no_grad():
        for _ in range(config.max_target_len - 1):
            decoder_input_ids = torch.tensor(
                [generated_ids],
                dtype=torch.long,
                device=device,
            )
            target_padding_mask = decoder_input_ids == plain_tokenizer.pad_token_id

            logits = model(
                src_ids=src_ids,
                decoder_input_ids=decoder_input_ids,
                src_padding_mask=src_padding_mask,
                target_padding_mask=target_padding_mask,
            )

            next_id = int(torch.argmax(logits[0, -1]).item())
            generated_ids.append(next_id)

            if next_id == plain_tokenizer.eos_token_id:
                break

    return plain_tokenizer.decode(generated_ids)


def write_predictions_csv(rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "index",
        "cipher_text",
        "target_plain_text",
        "predicted_plain_text",
        *METRIC_FIELDNAMES,
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "index": row["index"],
                    "cipher_text": row["cipher_text"],
                    "target_plain_text": row["target_plain_text"],
                    "predicted_plain_text": row["predicted_plain_text"],
                    **{
                        metric_name: row["metrics"].get(metric_name, "")
                        for metric_name in METRIC_FIELDNAMES
                    },
                }
            )


def evaluate_experiment(base_config, max_examples=100):
    checkpoint_dir = base_config.checkpoint_dir
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model, config, cipher_tokenizer, plain_tokenizer = load_checkpoint(
        checkpoint_dir=checkpoint_dir,
        device=device,
    )
    output_dir = os.path.join(config.output_dir, config.name.lower())

    dataset = CipherPlainDataset()
    _, _, test_dataset = create_datasplits(dataset, config)

    rows = []

    for index, item in enumerate(test_dataset):
        if index >= max_examples:
            break

        prediction = greedy_decode(
            model=model,
            cipher_text=item["cipher_text"],
            cipher_tokenizer=cipher_tokenizer,
            plain_tokenizer=plain_tokenizer,
            config=config,
            device=device,
        )

        rows.append(
            {
                "index": index,
                "cipher_text": item["cipher_text"],
                "target_plain_text": item["plain_text"],
                "predicted_plain_text": prediction,
                "metrics": compute_prediction_metrics(
                    prediction=prediction,
                    target=item["plain_text"],
                    tokenized_model=config.tokenizer_type != "blt",
                ),
            }
        )

    summary = average_metrics(rows)

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "evaluation_summary.json"), "w") as f:
        json.dump(
            {
                "config": asdict(config),
                "num_examples": len(rows),
                "metrics": summary,
            },
            f,
            indent=2,
        )

    write_predictions_csv(
        rows=rows,
        output_path=os.path.join(output_dir, "predictions.csv"),
    )

    metrics_path = os.path.join(checkpoint_dir, "metrics.json")
    if os.path.exists(metrics_path):
        training_metrics = load_json(metrics_path)
        plot_training_history(
            history=training_metrics.get("history", []),
            output_path=os.path.join(output_dir, "training_loss.png"),
        )

    plot_metric_summary(
        summary=summary,
        output_path=os.path.join(output_dir, "accuracy_overlap_metrics.png"),
    )
    plot_edit_distance(
        rows=rows,
        output_path=os.path.join(output_dir, "levenshtein_distance.png"),
    )
    plot_prediction_lengths(
        rows=rows,
        output_path=os.path.join(output_dir, "prediction_lengths.png"),
    )

    logger.info("Evaluation summary: %s", summary)
    logger.info("Saved report outputs to %s", output_dir)
    return {
        "name": config.name,
        **summary,
    }


def evaluate_c1(max_examples=100):
    return evaluate_experiment(EXPERIMENTS[0], max_examples=max_examples)


def evaluate_available_experiments(max_examples=100):
    experiment_rows = []

    for config in EXPERIMENTS:
        checkpoint_path = os.path.join(config.checkpoint_dir, "model.pt")

        if not os.path.exists(checkpoint_path):
            logger.info("Skipping %s because %s does not exist", config.name, checkpoint_path)
            continue

        experiment_rows.append(
            evaluate_experiment(
                base_config=config,
                max_examples=max_examples,
            )
        )

    if experiment_rows:
        output_path = os.path.join(EXPERIMENTS[0].output_dir, "ablation_summary.png")
        plot_ablation_summary(
            experiment_rows=experiment_rows,
            output_path=output_path,
        )

        with open(os.path.join(EXPERIMENTS[0].output_dir, "ablation_summary.json"), "w") as f:
            json.dump(experiment_rows, f, indent=2)

    return experiment_rows


def get_experiment_by_name(name):
    for config in EXPERIMENTS:
        if config.name.lower() == name.lower():
            return config

    supported = ", ".join(config.name.lower() for config in EXPERIMENTS)
    raise ValueError(f"Unknown experiment {name}. Supported values: {supported}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ANLP assignment experiments.")
    parser.add_argument(
        "--config",
        default="all",
        help="Experiment config to evaluate: C1, C2, C3, C4, C5, or all.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="Maximum number of test examples to decode per experiment.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()

    if args.config.lower() == "all":
        evaluate_available_experiments(max_examples=args.max_examples)
    else:
        evaluate_experiment(
            base_config=get_experiment_by_name(args.config),
            max_examples=args.max_examples,
        )
