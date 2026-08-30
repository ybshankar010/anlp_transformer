import math
from collections import Counter


def text_to_bits(text):
    return "".join(format(byte, "08b") for byte in text.encode("utf-8"))


def bit_level_accuracy(prediction, target):
    prediction_bits = text_to_bits(prediction)
    target_bits = text_to_bits(target)
    max_len = max(len(prediction_bits), len(target_bits))

    if max_len == 0:
        return 1.0

    matches = sum(
        pred_bit == target_bit
        for pred_bit, target_bit in zip(prediction_bits, target_bits)
    )
    return matches / max_len


def sequence_accuracy(prediction, target):
    return float(prediction == target)


def levenshtein_distance(source, target):
    if source == target:
        return 0

    if len(source) < len(target):
        source, target = target, source

    previous = list(range(len(target) + 1))

    for source_index, source_char in enumerate(source, start=1):
        current = [source_index]

        for target_index, target_char in enumerate(target, start=1):
            insert_cost = current[target_index - 1] + 1
            delete_cost = previous[target_index] + 1
            replace_cost = previous[target_index - 1] + (source_char != target_char)
            current.append(min(insert_cost, delete_cost, replace_cost))

        previous = current

    return previous[-1]


def tokenize_for_overlap(text):
    tokens = text.split()
    if tokens:
        return tokens
    return list(text)


def ngram_counts(tokens, n):
    if len(tokens) < n:
        return Counter()

    return Counter(
        tuple(tokens[index : index + n])
        for index in range(len(tokens) - n + 1)
    )


def clipped_ngram_precision(prediction_tokens, target_tokens, n):
    prediction_counts = ngram_counts(prediction_tokens, n)
    target_counts = ngram_counts(target_tokens, n)

    total = sum(prediction_counts.values())
    if total == 0:
        return 0.0

    overlap = prediction_counts & target_counts
    return sum(overlap.values()) / total


def bleu_score(prediction, target, max_n=4):
    prediction_tokens = tokenize_for_overlap(prediction)
    target_tokens = tokenize_for_overlap(target)

    if not prediction_tokens or not target_tokens:
        return 0.0

    effective_order = min(max_n, len(prediction_tokens), len(target_tokens))

    precisions = [
        clipped_ngram_precision(prediction_tokens, target_tokens, n)
        for n in range(1, effective_order + 1)
    ]

    smoothed_precisions = [
        precision if precision > 0 else 1e-9
        for precision in precisions
    ]

    log_precision = sum(math.log(precision) for precision in smoothed_precisions) / effective_order
    brevity_penalty = min(
        1.0,
        math.exp(1 - (len(target_tokens) / len(prediction_tokens))),
    )

    return brevity_penalty * math.exp(log_precision)


def rouge_n(prediction, target, n):
    prediction_tokens = tokenize_for_overlap(prediction)
    target_tokens = tokenize_for_overlap(target)

    prediction_counts = ngram_counts(prediction_tokens, n)
    target_counts = ngram_counts(target_tokens, n)

    total_target = sum(target_counts.values())
    if total_target == 0:
        return 0.0

    overlap = prediction_counts & target_counts
    return sum(overlap.values()) / total_target


def longest_common_subsequence_length(source_tokens, target_tokens):
    previous = [0] * (len(target_tokens) + 1)

    for source_token in source_tokens:
        current = [0]

        for target_index, target_token in enumerate(target_tokens, start=1):
            if source_token == target_token:
                current.append(previous[target_index - 1] + 1)
            else:
                current.append(max(previous[target_index], current[target_index - 1]))

        previous = current

    return previous[-1]


def rouge_l(prediction, target):
    prediction_tokens = tokenize_for_overlap(prediction)
    target_tokens = tokenize_for_overlap(target)

    if not target_tokens:
        return 0.0

    lcs_length = longest_common_subsequence_length(prediction_tokens, target_tokens)
    return lcs_length / len(target_tokens)


def compute_prediction_metrics(prediction, target, tokenized_model=True):
    metrics = {
        "bit_level_accuracy": bit_level_accuracy(prediction, target),
        "sequence_accuracy": sequence_accuracy(prediction, target),
        "levenshtein_distance": levenshtein_distance(prediction, target),
    }

    if tokenized_model:
        metrics.update(
            {
                "bleu": bleu_score(prediction, target),
                "rouge_1": rouge_n(prediction, target, 1),
                "rouge_2": rouge_n(prediction, target, 2),
                "rouge_l": rouge_l(prediction, target),
            }
        )

    return metrics


def average_metrics(rows):
    if not rows:
        return {}

    metric_names = rows[0]["metrics"].keys()

    return {
        metric_name: sum(row["metrics"][metric_name] for row in rows) / len(rows)
        for metric_name in metric_names
    }
