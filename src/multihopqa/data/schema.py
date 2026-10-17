"""Convert one raw Hugging Face row into the project's readable data shape."""

from multihopqa.data.alignment import make_sentence_nodes
from multihopqa.data.alignment import read_context
from multihopqa.data.alignment import read_supporting_facts


def normalize_example(raw_example):
    """Build one graph-ready example while retaining the source labels."""
    example_id = str(raw_example.get("id", ""))
    question = raw_example.get("question")
    answer = raw_example.get("answer")

    if not example_id:
        raise ValueError("A HotpotQA row has no id.")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Example {example_id} has an empty question.")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"Example {example_id} has an empty answer.")

    paragraphs = read_context(raw_example.get("context"))
    supporting_facts = read_supporting_facts(raw_example.get("supporting_facts"))
    sentence_nodes, support_records, unmatched_records = make_sentence_nodes(
        paragraphs,
        supporting_facts,
    )

    if answer.strip().casefold() in {"yes", "no"}:
        answer_kind = "yes_no"
    else:
        answer_kind = "span"

    return {
        "id": example_id,
        "question": question,
        "answer": answer,
        "answer_kind": answer_kind,
        "question_type": raw_example.get("type", "unknown"),
        "difficulty": raw_example.get("level", "unknown"),
        "sentences": sentence_nodes,
        "supporting_facts": support_records,
        "unmatched_supporting_facts": unmatched_records,
    }
