"""Compare graph topology and support reachability on a prepared split."""

import argparse
import json
from pathlib import Path

from multihopqa.graph.builder import GRAPH_MODES
from multihopqa.graph.builder import build_adjacency
from multihopqa.graph.diagnostics import graph_diagnostics
from multihopqa.utils.config import check_data_config
from multihopqa.utils.config import load_config


MEAN_FIELDS = (
    "edge_density",
    "mean_degree",
    "max_degree",
    "connected_components",
)


def analyze_split(file_path, graph_modes, limit=None):
    """Aggregate graph diagnostics for each requested construction mode."""
    totals = {
        graph_mode: {
            "examples": 0,
            "support_pair_count": 0,
            "support_pairs_within_1_hop": 0,
            "support_pairs_within_2_hops": 0,
            **{field: 0.0 for field in MEAN_FIELDS},
        }
        for graph_mode in graph_modes
    }
    examples_seen = 0

    with open(file_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            if limit is not None and examples_seen >= limit:
                break

            example = json.loads(line)
            nodes = example["sentences"]
            support_nodes = [
                node_index
                for node_index, node in enumerate(nodes)
                if node["is_supporting_fact"]
            ]

            for graph_mode in graph_modes:
                graph = build_adjacency(nodes, graph_mode=graph_mode)
                result = graph_diagnostics(graph, support_nodes)
                record = totals[graph_mode]
                record["examples"] += 1
                for field in MEAN_FIELDS:
                    record[field] += result[field]
                for field in (
                    "support_pair_count",
                    "support_pairs_within_1_hop",
                    "support_pairs_within_2_hops",
                ):
                    record[field] += result[field]
            examples_seen += 1

    results = {}
    for graph_mode, record in totals.items():
        example_count = record["examples"]
        if not example_count:
            raise ValueError(f"No examples were read from {file_path}.")

        summary = {
            "examples": example_count,
            "mean_edge_density": record["edge_density"] / example_count,
            "mean_degree": record["mean_degree"] / example_count,
            "mean_max_degree": record["max_degree"] / example_count,
            "mean_connected_components": (
                record["connected_components"] / example_count
            ),
        }
        pair_count = record["support_pair_count"]
        summary["support_pair_count"] = pair_count
        summary["support_pair_coverage_1_hop"] = (
            record["support_pairs_within_1_hop"] / pair_count if pair_count else 0.0
        )
        summary["support_pair_coverage_2_hops"] = (
            record["support_pairs_within_2_hops"] / pair_count if pair_count else 0.0
        )
        results[graph_mode] = summary
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-config",
        default="configs/data/hotpotqa_distractor.yaml",
        help="Dataset config path, relative to the repository folder.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Prepared split to analyze.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of examples to inspect.",
    )
    parser.add_argument(
        "--graph-modes",
        nargs="+",
        choices=GRAPH_MODES,
        default=list(GRAPH_MODES),
        help="Graph modes to compare.",
    )
    arguments = parser.parse_args()
    if arguments.limit is not None and arguments.limit < 1:
        parser.error("--limit must be greater than zero.")

    data_config = load_config(arguments.data_config)
    check_data_config(data_config)
    split_path = Path(data_config["output_dir"]) / f"{arguments.split}.jsonl"
    if not split_path.exists():
        raise FileNotFoundError(f"Prepared split not found: {split_path}")

    result = analyze_split(split_path, arguments.graph_modes, arguments.limit)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
