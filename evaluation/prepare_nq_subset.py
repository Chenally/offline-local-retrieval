"""Create a Natural Questions passage-retrieval subset (BEIR format).

Download and unzip the BEIR NQ package (corpus.jsonl, queries.jsonl,
qrels/test.tsv), for example from the BEIR datasets page.

Example:
  python evaluation/prepare_nq_subset.py \
      --beir-dir ~/datasets/beir/nq \
      --out evaluation/data/nq --num-queries 300 --num-distractors 2000
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from app.services.evaluation.datasets import prepare_nq_subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--beir-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("evaluation/data/nq"))
    parser.add_argument("--num-queries", type=int, default=300)
    parser.add_argument("--num-distractors", type=int, default=2000)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    queries, corpus = prepare_nq_subset(
        beir_dir=args.beir_dir.expanduser(),
        output_dir=args.out,
        num_queries=args.num_queries,
        num_distractors=args.num_distractors,
        split=args.split,
        seed=args.seed,
    )
    print(f"Wrote {queries} queries and {corpus} passages to {args.out}")


if __name__ == "__main__":
    main()
