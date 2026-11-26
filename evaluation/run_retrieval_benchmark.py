"""Index a prepared dataset with the real models and report retrieval quality.

Example:
  python evaluation/run_retrieval_benchmark.py --dataset evaluation/data/nq
  python evaluation/run_retrieval_benchmark.py --dataset evaluation/data/coco \
      --mode-set image_semantic --mode-set keyword,image_semantic

A fresh runtime directory is used for every run so earlier indexes never
leak into the measurement.
"""

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401
from app.config import AppSettings
from app.container import build_container
from app.services.evaluation.benchmark import run_benchmark

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--mode-set",
        action="append",
        help="Comma-separated modes, repeatable. Defaults to dataset.json.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    mode_sets = [value.split(",") for value in args.mode_set] if args.mode_set else None

    with tempfile.TemporaryDirectory(prefix="olr-benchmark-") as runtime:
        settings = replace(AppSettings.from_env(), runtime_dir=Path(runtime))
        container = build_container(settings)

        try:
            print("Model status:", container.component_status)
            report = run_benchmark(
                dataset_dir=args.dataset,
                indexing_service=container.indexing_service,
                retrieval_service=container.retrieval_service,
                mode_sets=mode_sets,
                top_k=args.top_k,
            )
        finally:
            container.close()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = args.output_name or report.dataset
    json_path = RESULTS_DIR / f"{name}.json"
    markdown_path = RESULTS_DIR / f"{name}.md"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(report.to_markdown(), encoding="utf-8")

    print(report.to_markdown())
    print(f"Saved {json_path} and {markdown_path}")


if __name__ == "__main__":
    main()
