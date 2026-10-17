"""Download and prepare the complete HotpotQA distractor train/validation data."""

import argparse

from multihopqa.data.hotpotqa import prepare_hotpotqa
from multihopqa.utils.config import check_data_config
from multihopqa.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Prepare HotpotQA from Hugging Face.")
    parser.add_argument(
        "--config",
        default="configs/data/hotpotqa_distractor.yaml",
        help="Path to the data YAML file, relative to the repository folder.",
    )
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    check_data_config(config)
    prepare_hotpotqa(config)


if __name__ == "__main__":
    main()
