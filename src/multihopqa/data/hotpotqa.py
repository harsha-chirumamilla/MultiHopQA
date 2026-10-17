"""Download HotpotQA from Hugging Face and save graph-ready JSONL files."""

import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from multihopqa.data.schema import normalize_example


def prepare_hotpotqa(config):
    """Prepare every configured split and write a short preparation summary."""
    cache_dir = Path(config["cache_dir"])
    output_dir = Path(config["output_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset_id": config["dataset_id"],
        "dataset_config": config["dataset_config"],
        "splits": {},
    }

    for split_name in config["splits"]:
        print(f"Loading {split_name} split from Hugging Face...")
        dataset = load_dataset(
            config["dataset_id"],
            config["dataset_config"],
            split=split_name,
            cache_dir=str(cache_dir),
        )

        output_path = output_dir / f"{split_name}.jsonl"
        split_stats = write_split(dataset, split_name, output_path)
        summary["splits"][split_name] = split_stats
        print(f"Saved {split_stats['examples']} examples to {output_path}")

    summary_path = output_dir / "preparation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)
        summary_file.write("\n")

    print(f"Preparation summary: {summary_path}")
    return summary


def write_split(dataset, split_name, output_path):
    """Normalize and write one split, one JSON object per line."""
    answer_kinds = Counter()
    question_types = Counter()
    examples = 0
    sentence_count = 0
    supporting_sentence_count = 0
    unmatched_supporting_fact_count = 0
    examples_with_unmatched_supporting_facts = 0
    unmatched_examples = []

    temporary_path = output_path.with_name(output_path.name + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        for raw_example in tqdm(dataset, desc=f"Preparing {split_name}", unit="example"):
            example = normalize_example(raw_example)
            output_file.write(json.dumps(example, ensure_ascii=False) + "\n")

            examples += 1
            sentence_count += len(example["sentences"])
            supporting_sentence_count += sum(
                sentence["is_supporting_fact"] for sentence in example["sentences"]
            )
            unmatched_facts = example["unmatched_supporting_facts"]
            unmatched_supporting_fact_count += len(unmatched_facts)
            if unmatched_facts:
                examples_with_unmatched_supporting_facts += 1
                if len(unmatched_examples) < 20:
                    unmatched_examples.append(
                        {"id": example["id"], "facts": unmatched_facts}
                    )
            answer_kinds[example["answer_kind"]] += 1
            question_types[str(example["question_type"])] += 1

    temporary_path.replace(output_path)

    return {
        "examples": examples,
        "sentences": sentence_count,
        "supporting_sentences": supporting_sentence_count,
        "unmatched_supporting_facts": unmatched_supporting_fact_count,
        "examples_with_unmatched_supporting_facts": examples_with_unmatched_supporting_facts,
        "unmatched_examples_sample": unmatched_examples,
        "answer_kinds": dict(answer_kinds),
        "question_types": dict(question_types),
        "file": str(output_path),
    }
