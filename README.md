# ANLP Assignment 1: Custom Transformer for Cipher-to-Plaintext Reconstruction

This project implements a custom sequence-to-sequence Transformer for mapping encrypted
binary sequences to plaintext English. It was built for Advanced Natural Language
Processing Assignment 1.

The source sequence is the encrypted binary text from `brown_cipher.txt`, and the target
sequence is the aligned plaintext sentence from `brown_plain.txt`.

## Assignment Requirements Covered

- Encoder-decoder Transformer implemented using PyTorch modules and tensor operations.
- No use of `torch.nn.Transformer`.
- No use of `torch.nn.MultiheadAttention`.
- Scratch scaled dot-product attention, multi-head attention, grouped-query attention,
  feed-forward network, normalization, encoder, and decoder blocks.
- Scratch BPE-style tokenizer for C1-C4.
- Simplified BLT-style token-free path for C5.
- Training logs through Weights & Biases.
- Model checkpoints uploaded to Hugging Face.
- Greedy-decoding evaluation with the required assignment metrics.

## Architecture Overview

The project is organized around a shared training and evaluation pipeline.

- `src/dataset.py`
  - Loads aligned cipher/plaintext pairs.
  - Provides the BPE-based collator for C1-C4.
  - Provides the byte/token-free collator for C5.

- `src/tokenizer.py`
  - Implements a scratch BPE-style tokenizer.
  - Saves and loads tokenizer state for reproducible checkpoint evaluation.

- `src/models/attention.py`
  - Implements scaled dot-product attention.
  - Implements multi-head attention.
  - Implements grouped-query attention.
  - Applies RoPE rotation when selected by configuration.

- `src/models/positional.py`
  - Implements sinusoidal absolute positional encoding.

- `src/models/norm.py`
  - Implements scratch LayerNorm.
  - Implements RMSNorm.

- `src/models/encoder.py` and `src/models/decoder.py`
  - Build reusable encoder and decoder stacks.

- `src/models/transformer.py`
  - Defines the C1-C4 sequence-to-sequence Transformer.

- `src/models/blt.py`
  - Defines the simplified C5 BLT-style model.
  - Uses local bit/byte processing with fixed-size source patches.

- `src/train.py`
  - Trains a selected experiment configuration.
  - Logs training and validation loss to W&B.
  - Saves checkpoints and uploads them to Hugging Face.

- `src/evaluate.py`
  - Loads saved checkpoints.
  - Runs greedy decoding.
  - Computes required metrics.
  - Writes local outputs and logs aggregate metrics to W&B.

## Experiment Configurations

| Config | Change from Base | Positional Encoding | Attention | Normalization | Tokenization |
| --- | --- | --- | --- | --- | --- |
| C1 | Base | Sinusoidal absolute | MHA | LayerNorm | Scratch BPE subword |
| C2 | Positional encoding | RoPE | MHA | LayerNorm | Scratch BPE subword |
| C3 | Attention mechanism | Sinusoidal absolute | GQA | LayerNorm | Scratch BPE subword |
| C4 | Normalization | Sinusoidal absolute | MHA | RMSNorm | Scratch BPE subword |
| C5 | Tokenization | Sinusoidal absolute | MHA | LayerNorm | BLT-style token-free |

## Setup

This project uses Python 3.12 and `uv`.

Install dependencies:

```bash
uv sync
```

Create a `.env` file in the project root:

```env
WANDB_API_KEY=your_wandb_api_key
HF_TOKEN=your_huggingface_token
```

The dataset should be available at:

```text
Dataset_A1/brown_cipher.txt
Dataset_A1/brown_plain.txt
```

## Run Training

Train a single configuration:

```bash
python3 -m src.train --config C1
```

Other configurations:

```bash
python3 -m src.train --config C2
python3 -m src.train --config C3
python3 -m src.train --config C4
python3 -m src.train --config C5
```

Run all configurations overnight:

```bash
bash run_overnight.sh
```

Useful overnight variants:

```bash
CONFIGS="C1 C2 C3" bash run_overnight.sh
MAX_EXAMPLES=500 bash run_overnight.sh
SKIP_EXISTING=1 bash run_overnight.sh
```

## Run Evaluation

Evaluate one completed configuration:

```bash
python3 -m src.evaluate --config C1
```

Evaluate all available checkpoints:

```bash
python3 -m src.evaluate --config all
```

Run local evaluation without logging metrics to W&B:

```bash
python3 -m src.evaluate --config C1 --no-wandb
```

Evaluation uses greedy decoding so that results are consistent across configurations.

## Metrics

The evaluation script computes the assignment metrics:

- Bit-level accuracy
- Sequence accuracy
- Levenshtein distance
- BLEU
- ROUGE-1
- ROUGE-2
- ROUGE-L

For C1-C4, BLEU and ROUGE are computed over tokenized text-style outputs. C5 is the
token-free BLT-style configuration and is evaluated primarily through reconstruction
metrics.

## Outputs and Links

Local checkpoint folders:

```text
checkpoints/c1/
checkpoints/c2/
checkpoints/c3/
checkpoints/c4/
checkpoints/c5/
```

Local evaluation outputs:

```text
outputs/c1/
outputs/c2/
outputs/c3/
outputs/c4/
outputs/c5/
outputs/run_logs/
```

Each evaluated configuration writes:

```text
evaluation_summary.json
predictions.csv
training_loss.png
accuracy_overlap_metrics.png
levenshtein_distance.png
prediction_lengths.png
```

The combined ablation evaluation writes:

```text
outputs/ablation_summary.json
outputs/ablation_summary.png
```

Hugging Face checkpoint repository:

- [https://huggingface.co/ybs010/anlp-assignment1/tree/main/c1](https://huggingface.co/ybs010/anlp-assignment1/tree/main/c1)

Weights & Biases project:

- [https://wandb.ai/ybs010-iiit-hyderabad/anlp-assignment1](https://wandb.ai/ybs010-iiit-hyderabad/anlp-assignment1)

## Repository Structure

```text
2026900001_assignment1/
|-- src/
|   |-- models/
|   |   |-- attention.py
|   |   |-- positional.py
|   |   |-- norm.py
|   |   |-- embedding.py
|   |   |-- feedforward.py
|   |   |-- encoder.py
|   |   |-- decoder.py
|   |   |-- transformer.py
|   |   `-- blt.py
|   |-- constants.py
|   |-- dataset.py
|   |-- evaluate.py
|   |-- metrics.py
|   |-- plots.py
|   |-- tokenizer.py
|   |-- train.py
|   `-- utils.py
|-- Dataset_A1/
|   |-- brown_cipher.txt
|   `-- brown_plain.txt
|-- outputs/
|-- checkpoints/
|-- run_overnight.sh
|-- pyproject.toml
|-- uv.lock
`-- README.md
```
