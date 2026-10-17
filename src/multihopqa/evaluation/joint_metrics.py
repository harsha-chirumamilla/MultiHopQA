"""Combine answer and supporting-fact scores into joint QA scores."""


def joint_scores(answer, support):
    """Return joint exact match and joint F1 from two score tuples."""
    answer_em, answer_precision, answer_recall, _ = answer
    support_em, support_precision, support_recall, _ = support

    joint_em = answer_em * support_em
    joint_precision = answer_precision * support_precision
    joint_recall = answer_recall * support_recall

    if joint_precision + joint_recall == 0:
        joint_f1 = 0.0
    else:
        joint_f1 = (
            2.0 * joint_precision * joint_recall
            / (joint_precision + joint_recall)
        )

    return joint_em, joint_f1
