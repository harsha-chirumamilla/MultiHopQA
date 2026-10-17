"""Answer exact-match and token F1 metrics."""

import re
import string
from collections import Counter


def normalize_answer(answer):
    """Apply the usual lowercase, punctuation, article, and whitespace rules."""
    answer = answer.casefold()
    answer = "".join(character for character in answer if character not in string.punctuation)
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    return " ".join(answer.split())


def answer_scores(prediction, gold_answer):
    """Return exact match, precision, recall, and token F1 for one answer."""
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold_answer)
    prediction_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()

    exact_match = float(normalized_prediction == normalized_gold)

    if not prediction_tokens and not gold_tokens:
        return exact_match, 1.0, 1.0, 1.0
    if not prediction_tokens or not gold_tokens:
        return exact_match, 0.0, 0.0, 0.0

    common_count = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
    if common_count == 0:
        return exact_match, 0.0, 0.0, 0.0

    precision = common_count / len(prediction_tokens)
    recall = common_count / len(gold_tokens)
    f1 = 2.0 * precision * recall / (precision + recall)
    return exact_match, precision, recall, f1
