import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "anlp_assignment1_matplotlib"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    os.path.join(tempfile.gettempdir(), "anlp_assignment1_cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def ensure_parent_dir(path):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def plot_training_history(history, output_path):
    if not history:
        return

    ensure_parent_dir(output_path)

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    validation_loss = [row["validation_loss"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="Train loss")
    plt.plot(epochs, validation_loss, marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("C1 Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_metric_summary(summary, output_path):
    if not summary:
        return

    ensure_parent_dir(output_path)

    metric_names = [
        name
        for name, value in summary.items()
        if 0.0 <= value <= 1.0
    ]
    values = [summary[name] for name in metric_names]

    plt.figure(figsize=(9, 5))
    plt.bar(metric_names, values)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("C1 Accuracy and Overlap Metrics")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_edit_distance(rows, output_path):
    if not rows:
        return

    ensure_parent_dir(output_path)

    distances = [row["metrics"]["levenshtein_distance"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.hist(distances, bins=20)
    plt.xlabel("Levenshtein distance")
    plt.ylabel("Number of examples")
    plt.title("C1 Levenshtein Distance Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_ablation_summary(experiment_rows, output_path):
    if not experiment_rows:
        return

    ensure_parent_dir(output_path)

    names = [row["name"] for row in experiment_rows]
    bit_accuracy = [row["bit_level_accuracy"] for row in experiment_rows]
    sequence_accuracy = [row["sequence_accuracy"] for row in experiment_rows]
    bleu = [row.get("bleu", 0.0) for row in experiment_rows]
    rouge_l = [row.get("rouge_l", 0.0) for row in experiment_rows]

    x_positions = range(len(names))
    width = 0.2

    plt.figure(figsize=(10, 5))
    plt.bar(
        [x - 1.5 * width for x in x_positions],
        bit_accuracy,
        width=width,
        label="Bit accuracy",
    )
    plt.bar(
        [x - 0.5 * width for x in x_positions],
        sequence_accuracy,
        width=width,
        label="Sequence accuracy",
    )
    plt.bar(
        [x + 0.5 * width for x in x_positions],
        bleu,
        width=width,
        label="BLEU",
    )
    plt.bar(
        [x + 1.5 * width for x in x_positions],
        rouge_l,
        width=width,
        label="ROUGE-L",
    )
    plt.xticks(list(x_positions), names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Ablation Summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_prediction_lengths(rows, output_path):
    if not rows:
        return

    ensure_parent_dir(output_path)

    target_lengths = [len(row["target_plain_text"]) for row in rows]
    prediction_lengths = [len(row["predicted_plain_text"]) for row in rows]

    plt.figure(figsize=(6, 6))
    plt.scatter(target_lengths, prediction_lengths, alpha=0.65)
    max_len = max(target_lengths + prediction_lengths)
    plt.plot([0, max_len], [0, max_len], linestyle="--", color="black", linewidth=1)
    plt.xlabel("Target plaintext length")
    plt.ylabel("Predicted plaintext length")
    plt.title("C1 Prediction Lengths")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
