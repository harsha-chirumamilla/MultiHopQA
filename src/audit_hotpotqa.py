"""Check the prepared JSONL files for basic data and label problems."""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from multihopqa.utils.config import check_data_config
from multihopqa.utils.config import load_config


def audit_split(file_path):
    """Read a prepared split and verify node labels and example identifiers."""
    seen_ids = set()
    examples = 0
    sentences = 0
    supporting_sentences = 0
    problems = []
    unmatched_supporting_facts = 0
    examples_with_unmatched_supporting_facts = 0

    with open(file_path, "r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(
            tqdm(input_file, desc=f"Auditing {file_path.name}"), start=1
        ):
            example = json.loads(line)
            example_id = example.get("id")

            if not example_id:
                problems.append(f"line {line_number}: missing id")
            elif example_id in seen_ids:
                problems.append(f"line {line_number}: duplicate id {example_id}")
            else:
                seen_ids.add(example_id)

            if not example.get("question"):
                problems.append(f"line {line_number}: missing question")
            if not example.get("answer"):
                problems.append(f"line {line_number}: missing answer")

            nodes = example.get("sentences", [])
            support_keys = {
                (fact.get("title"), fact.get("sentence_index"))
                for fact in example.get("supporting_facts", [])
            }
            unmatched_keys = {
                (fact.get("title"), fact.get("sentence_index"))
                for fact in example.get("unmatched_supporting_facts", [])
            }
            aligned_support_keys = support_keys - unmatched_keys
            node_keys = set()

            for node_index, node in enumerate(nodes):
                node_key = (node.get("title"), node.get("sentence_index"))
                node_keys.add(node_key)
                if node.get("node_index") != node_index:
                    problems.append(f"line {line_number}: sentence node indexes are not sequential")
                if node.get("is_supporting_fact") != (node_key in aligned_support_keys):
                    problems.append(f"line {line_number}: supporting-fact label is misaligned")

            if aligned_support_keys - node_keys:
                problems.append(f"line {line_number}: supporting fact is missing from sentence nodes")

            if unmatched_keys & node_keys:
                problems.append(f"line {line_number}: a matched support fact was marked unmatched")
            unmatched_supporting_facts += len(unmatched_keys)
            if unmatched_keys:
                examples_with_unmatched_supporting_facts += 1

            examples += 1
            sentences += len(nodes)
            supporting_sentences += sum(
                node.get("is_supporting_fact", False) for node in nodes
            )

    return {
        "file": str(file_path),
        "examples": examples,
        "sentences": sentences,
        "supporting_sentences": supporting_sentences,
        "unmatched_supporting_facts": unmatched_supporting_facts,
        "examples_with_unmatched_supporting_facts": examples_with_unmatched_supporting_facts,
        "problems": problems,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit prepared HotpotQA files.")
    parser.add_argument(
        "--config",
        default="configs/data/hotpotqa_distractor.yaml",
        help="Path to the data YAML file, relative to the repository folder.",
    )
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    check_data_config(config)
    output_dir = Path(config["output_dir"])
    all_passed = True

    for split_name in config["splits"]:
        file_path = output_dir / f"{split_name}.jsonl"
        if not file_path.exists():
            raise FileNotFoundError(
                f"Prepared file not found: {file_path}. Run src/prepare_hotpotqa.py first."
            )

        result = audit_split(file_path)
        print(
            f"{split_name}: {result['examples']} examples, "
            f"{result['sentences']} sentence nodes, "
            f"{result['supporting_sentences']} supporting sentences"
        )
        if result["unmatched_supporting_facts"]:
            print(
                "  WARNING: "
                f"{result['unmatched_supporting_facts']} source supporting-fact references "
                "do not match a sentence in the supplied context "
                f"({result['examples_with_unmatched_supporting_facts']} examples). "
                "They are retained in the output and excluded from node labels."
            )

        if result["problems"]:
            all_passed = False
            for problem in result["problems"][:10]:
                print(f"  PROBLEM: {problem}")
            if len(result["problems"]) > 10:
                print(f"  ... and {len(result['problems']) - 10} more problems")
        else:
            print("  Checks passed")

    if not all_passed:
        raise SystemExit("HotpotQA audit found data problems.")


if __name__ == "__main__":
    main()
