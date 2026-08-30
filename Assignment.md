# Advanced NLP Assignment 1 - Working Guide

## Objective

Build an Encoder-Decoder Transformer from scratch in PyTorch for mapping encrypted
binary sequences to plaintext.

Do not use:
- nn.Transformer
- nn.MultiheadAttention
- prebuilt tokenizers from libraries

The purpose is not merely to achieve high accuracy.
The goal is to understand, implement, and experimentally compare Transformer
architectural choices.

The student must be able to explain all code during a viva.

## Interaction Rule

Do NOT write complete solutions for me unless I explicitly ask.

Guide me toward implementation.

When reviewing my code:
1. Explain what is wrong.
2. Explain the underlying concept.
3. Give hints about how to fix it.
4. Let me attempt the implementation.
5. Review my attempt.

Prefer discussing tensor shapes and data flow.

---

# Assignment Configurations

## C1 - Baseline
- Sinusoidal absolute positional encoding
- Multi-Head Attention
- LayerNorm
- Learned subword tokenization implemented from scratch

## C2
Change only:
- Sinusoidal positional encoding -> RoPE

## C3
Change only:
- MHA -> Grouped Query Attention

## C4
Change only:
- LayerNorm -> RMSNorm

## C5
Change only tokenization/model input:
- Scratch BPE-style tokenizer -> simplified Byte Latent Transformer
- Raw bytes
- Local encoder
- Byte patches
- Global Transformer
- Local decoder

Keep other hyperparameters consistent wherever possible.

---

# Overall Implementation Strategy

## Phase 1 - Data + Experiment Infrastructure

Goal:
Understand the dataset and establish the reusable training/evaluation pipeline.

### 1A - Dataset EDA

Dataset consists of two line-aligned files:

brown_cipher.txt
brown_plain.txt

Line N in cipher corresponds to line N in plaintext.

Before creating the PyTorch Dataset, determine:

- number of cipher examples
- number of plaintext examples
- whether counts match
- empty lines
- unique characters in cipher
- cipher sequence length:
  - min
  - max
  - mean
  - median
- plaintext character length:
  - min
  - max
  - mean
  - median
- inspect 2-3 aligned examples

Do NOT choose tokenization until this analysis is complete.

### 1B - Dataset Design

After EDA, keep the raw aligned dataset separate from tokenization.

Each raw example should preserve:

ciphertext:
raw encrypted binary string

plaintext:
raw English text string

Understand:
- source text
- target text
- line alignment
- why splitting must happen at the pair level

### 1C - Data Split

Create reproducible:
- train
- validation
- test

Use a fixed random seed.

Split the raw aligned pairs before tokenizer training.

Do not train tokenizers on validation or test examples.

### 1D - Scratch BPE Tokenizers

For C1-C4, implement learned subword tokenization from scratch.

Do not use:
- Hugging Face tokenizers
- SentencePiece library
- tokenizers library
- pretrained tokenizers
- fixed-width chunking such as 8 bits = 1 token

Fixed-width bit grouping does not count as learned subword tokenization.

Recommended tokenizer:
BPE-style tokenizer learned from adjacent pair frequencies.

Train two tokenizer instances:

cipher tokenizer:
training ciphertext only
-> learned binary subword units
-> source token IDs

plaintext tokenizer:
training plaintext only
-> learned text subword units
-> target token IDs

Both tokenizers should support:
- special tokens such as PAD, UNK, BOS, EOS
- train(texts)
- encode(text)
- decode(ids)
- saved vocabulary / merge rules if practical

For ciphertext, initial tokens may be individual characters:
- "0"
- "1"

Learned merge examples may become variable-length tokens such as:
- "01"
- "110"
- "00101"

This is acceptable because token boundaries are learned from data frequency.

This is not acceptable:
- always grouping every 4 bits
- always grouping every 8 bits
- always grouping every N bits

### 1E - Tensor Dataset + Collation

After train/validation/test split and tokenizer training, convert examples into
model-ready tensors:

cipher:
raw sequence
-> scratch BPE tokenizer
-> source token IDs
-> padding/masking
-> tensor

plaintext:
sentence
-> scratch BPE tokenizer
-> target token IDs
-> BOS/EOS
-> padding/masking
-> tensor

Understand:
- source sequence
- target sequence
- decoder input
- decoder target

For decoder training:

plaintext token IDs:
target_ids

decoder input:
BOS + target_ids

decoder target:
target_ids + EOS

Batch collation should produce:
- source input IDs
- source padding mask
- decoder input IDs
- decoder target IDs
- target padding mask

### 1F - Experiment Configuration

Create one configuration mechanism containing things such as:

- model dimension
- number of heads
- encoder layers
- decoder layers
- FFN dimension
- dropout
- learning rate
- batch size
- maximum sequence lengths
- positional encoding type
- attention type
- normalization type
- tokenizer type
- source vocabulary size
- target vocabulary size
- BPE merge count or vocabulary limit
- minimum merge frequency if used

C1-C5 should primarily be controlled through configuration rather than duplicated code.

### 1G - Metrics

Eventually support:

- bit-level accuracy
- sequence accuracy
- Levenshtein distance
- BLEU
- ROUGE

Use greedy decoding for evaluation.

### 1H - Experiment Logging

Use Weights & Biases.

Every run should record:

identity:
- configuration C1-C5
- hyperparameters
- random seed

training:
- training loss
- validation loss
- epoch
- global step
- learning rate

tokenization:
- source tokenizer type
- target tokenizer type
- source vocabulary size
- target vocabulary size
- number of learned merges
- sequence lengths after tokenization

performance:
- epoch duration / training time
- examples/sec or tokens/sec if practical
- peak GPU memory

evaluation:
- bit accuracy
- sequence accuracy
- Levenshtein distance
- BLEU
- ROUGE where applicable

Artifacts:
- checkpoints
- evaluation predictions
- final metrics

Trained checkpoints must eventually be uploaded to Hugging Face.

---

# Phase 2 - C1 Baseline Transformer

Do NOT try to implement the complete Transformer at once.

Build modules incrementally.

Recommended order:

1. embeddings
2. sinusoidal positional encoding
3. scaled dot-product attention
4. Multi-Head Attention
5. position-wise FFN
6. LayerNorm + residual connections
7. encoder layer
8. encoder stack
9. decoder self-attention
10. causal mask
11. encoder-decoder cross attention
12. decoder layer
13. decoder stack
14. output projection
15. complete Seq2Seq Transformer
16. loss
17. greedy decoding

First important milestone:

one batch
-> encoder
-> decoder
-> logits
-> loss

Before long training, intentionally overfit a very small number of examples
to verify that the implementation can learn.

Then train C1 and collect all experiment logs.

---

# Phase 3 - Architectural Ablations

Start from the working C1 implementation.

## C2 - RoPE

Understand:
- what positional information Q/K need
- how rotation introduces relative position information

Replace only positional encoding behavior required for C2.

Run through exactly the same training/evaluation pipeline.

## C3 - GQA

Understand:
- query heads
- key/value heads
- how multiple query heads share fewer K/V heads

Replace only the attention mechanism.

Run through the same pipeline.

## C4 - RMSNorm

Understand:
- LayerNorm centers and scales
- RMSNorm scales without mean-centering

Replace only normalization.

Run through the same pipeline.

---

# Phase 4 - BLT / C5

Do this last.

Understand the conceptual pipeline before implementation:

raw bytes
-> local byte encoder
-> patches
-> global Transformer
-> local byte decoder
-> output bytes

Compare especially against C1 on:

- training speed
- computational overhead
- peak GPU memory
- reconstruction performance

---

# Experimental Principle

This is an ablation study.

Only ONE main architectural choice should change relative to C1.

Avoid accidentally changing things such as:
- learning rate
- model width
- model depth
- train/test split
- batch size

unless technically necessary and explicitly documented.

---

# Report Evidence

Collect information while implementing.

Do not wait until the end to instrument experiments.

Final report will likely need:

- C1-C5 final metric comparison table
- training/validation loss curves
- training speed comparison
- GPU memory comparison
- reconstruction performance comparison
- discussion of why each variant improved or worsened

Do not spend time polishing plots until experiments are complete.

---

# Viva Preparation

50% of marks are for code defense.

For every module I implement, make sure I can answer:

- What goes into this module?
- What comes out?
- What are the tensor shapes?
- What parameters are learned?
- Why does this component exist?
- What would happen if it were removed?
- How is this implementation different in C1-C5?

When reviewing my implementation, actively question me about these topics.
