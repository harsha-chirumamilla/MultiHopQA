"""Supporting-fact exact-match, precision, recall, and F1."""


def support_scores(predicted_facts, gold_facts):
    """Compare sets of (article title, sentence index) pairs."""
    predicted = {
        (fact["title"], int(fact["sentence_index"]))
        for fact in predicted_facts
    }
    gold = {
        (fact["title"], int(fact["sentence_index"]))
        for fact in gold_facts
    }

    exact_match = float(predicted == gold)
    correct = len(predicted & gold)

    if not predicted and not gold:
        return exact_match, 1.0, 1.0, 1.0
    if not predicted or not gold:
        return exact_match, 0.0, 0.0, 0.0

    precision = correct / len(predicted)
    recall = correct / len(gold)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return exact_match, precision, recall, f1
