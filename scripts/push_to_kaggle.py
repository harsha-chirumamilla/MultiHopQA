#!/usr/bin/env python3
"""Package this project as a Kaggle script kernel and start a training run."""

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KERNEL_CODE_FILE = "run_training.py"


def project_files():
    """Return only the source, configs, and dependency list needed to train."""
    files = [PROJECT_ROOT / "requirements.txt"]
    for source_dir in (PROJECT_ROOT / "src", PROJECT_ROOT / "configs"):
        files.extend(
            path
            for path in source_dir.rglob("*")
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}
        )
    return sorted(files)


def make_project_archive():
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in project_files():
            bundle.write(path, path.relative_to(PROJECT_ROOT).as_posix())
    return base64.b64encode(archive.getvalue()).decode("ascii")


def make_kernel_runner(archive_b64, train_device="xla"):
    encoded_lines = textwrap.wrap(archive_b64, width=100)
    archive_literal = "\n".join(f'    "{line}"' for line in encoded_lines)
    runner = '''\
        """Bootstrap and run the packaged MultihopQA trainer on Kaggle."""

        import base64
        import io
        import os
        from pathlib import Path
        import subprocess
        import sys
        import zipfile

        TRAIN_DEVICE = "__TRAIN_DEVICE__"
        PROJECT_ARCHIVE = (
            __ARCHIVE_LITERAL__
        )

        def main():
            working_dir = Path("/kaggle/working")
            if not working_dir.is_dir():
                working_dir = Path.cwd()
            project_dir = working_dir / "multihopqa_project"
            project_dir.mkdir(parents=True, exist_ok=True)

            archive_data = base64.b64decode(PROJECT_ARCHIVE)
            with zipfile.ZipFile(io.BytesIO(archive_data)) as project_archive:
                project_archive.extractall(project_dir)

            os.chdir(project_dir)
            sys.path.insert(0, str(project_dir / "src"))

            try:
                import datasets  # noqa: F401
                import torch  # noqa: F401
                import transformers  # noqa: F401
                import yaml  # noqa: F401
                import tqdm  # noqa: F401
            except (ImportError, AttributeError):
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        str(project_dir / "requirements.txt"),
                    ],
                    check=True,
                )

            if TRAIN_DEVICE == "xla":
                try:
                    import torch_xla  # noqa: F401
                except ImportError as error:
                    raise RuntimeError(
                        "This Kaggle run was configured for TPU, but its image "
                        "does not contain torch_xla. Kaggle may not have assigned "
                        "the requested TPU accelerator to this pushed kernel."
                    ) from error

            import yaml

            data_config_path = project_dir / "configs/data/hotpotqa_distractor.yaml"
            with data_config_path.open(encoding="utf-8") as config_file:
                data_config = yaml.safe_load(config_file)
            data_dir = Path(data_config["output_dir"])
            if not data_dir.is_absolute():
                data_dir = project_dir / data_dir
            required_splits = (data_dir / "train.jsonl", data_dir / "validation.jsonl")
            if not all(path.is_file() for path in required_splits):
                print("Prepared HotpotQA splits were not found; preparing them now.")
                subprocess.run(
                    [sys.executable, "src/prepare_hotpotqa.py"],
                    cwd=project_dir,
                    check=True,
                )

            print(f"Training project files are in {project_dir}")
            subprocess.run(
                [sys.executable, "src/train.py", "--device", TRAIN_DEVICE],
                cwd=project_dir,
                check=True,
            )
            print(f"Training artifacts are under {project_dir / 'outputs'}")


        if __name__ == "__main__":
            main()
    '''
    runner = textwrap.dedent(runner)
    runner = runner.replace("__ARCHIVE_LITERAL__", archive_literal)
    return runner.replace("__TRAIN_DEVICE__", train_device)


def title_from_kernel_id(kernel_id):
    slug = kernel_id.split("/", maxsplit=1)[1]
    return " ".join(part.capitalize() for part in slug.replace("_", "-").split("-"))


def write_kernel_bundle(directory, kernel_id, title, is_private, accelerator="tpu"):
    if accelerator not in {"gpu", "cpu", "tpu"}:
        raise ValueError(f"Unsupported Kaggle accelerator: {accelerator}")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / KERNEL_CODE_FILE).write_text(
        make_kernel_runner(
            make_project_archive(),
            train_device="xla" if accelerator == "tpu" else "auto",
        ),
        encoding="utf-8",
    )
    metadata = {
        "id": kernel_id,
        "title": title,
        "code_file": KERNEL_CODE_FILE,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true" if is_private else "false",
        "enable_gpu": "true" if accelerator == "gpu" else "false",
        "enable_tpu": "true" if accelerator == "tpu" else "false",
        "enable_internet": "true",
        "machine_shape": "TpuV5E8" if accelerator == "tpu" else "",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (directory / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def build_push_command(staging_dir, accelerator):
    """Build the Kaggle CLI command for the selected accelerator."""
    command = ["kaggle", "kernels", "push", "-p", str(staging_dir)]
    if accelerator == "tpu":
        command.extend(["--accelerator", "TpuV5E8"])
    return command


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Push this project as a Kaggle script kernel. Kaggle starts the trainer "
            "after the upload completes."
        )
    )
    parser.add_argument(
        "--kernel-id",
        default=os.environ.get("KAGGLE_KERNEL_ID"),
        help="Kaggle owner/slug (or set KAGGLE_KERNEL_ID).",
    )
    parser.add_argument(
        "--title",
        help="Kernel title. Defaults to a title derived from the slug.",
    )
    accelerator_group = parser.add_mutually_exclusive_group()
    accelerator_group.add_argument(
        "--cpu",
        action="store_true",
        help="Request a CPU kernel instead of the default Kaggle TPU.",
    )
    accelerator_group.add_argument(
        "--tpu",
        action="store_true",
        help="Request a Kaggle TPU v5e-8 kernel (the default).",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the Kaggle kernel public. Kernels are private by default.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        help="Keep the generated upload files in this directory for inspection.",
    )
    arguments = parser.parse_args()
    if not arguments.kernel_id or arguments.kernel_id.count("/") != 1:
        parser.error("--kernel-id must be Kaggle's owner/slug, or set KAGGLE_KERNEL_ID")
    owner, slug = arguments.kernel_id.split("/", maxsplit=1)
    if not owner or not slug:
        parser.error("--kernel-id must include both a Kaggle owner and a kernel slug")
    if shutil.which("kaggle") is None:
        parser.error("Kaggle CLI was not found; install and configure the Kaggle API first")
    return arguments


def main():
    arguments = parse_args()
    title = arguments.title or title_from_kernel_id(arguments.kernel_id)
    accelerator = "cpu" if arguments.cpu else "tpu"

    if arguments.staging_dir:
        staging_dir = arguments.staging_dir.resolve()
        write_kernel_bundle(
            staging_dir,
            arguments.kernel_id,
            title,
            is_private=not arguments.public,
            accelerator=accelerator,
        )
        subprocess.run(build_push_command(staging_dir, accelerator), check=True)
    else:
        with tempfile.TemporaryDirectory(prefix="multihopqa-kaggle-") as temporary_dir:
            staging_dir = Path(temporary_dir)
            write_kernel_bundle(
                staging_dir,
                arguments.kernel_id,
                title,
                is_private=not arguments.public,
                accelerator=accelerator,
            )
            subprocess.run(build_push_command(staging_dir, accelerator), check=True)

    print(f"Pushed {arguments.kernel_id}; Kaggle will run the trainer.")


if __name__ == "__main__":
    main()
