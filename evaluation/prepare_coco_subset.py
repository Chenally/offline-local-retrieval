"""Create a COCO text-to-image retrieval subset.

Download from https://cocodataset.org/#download:
  - 2017 Val images        -> val2017/
  - 2017 Train/Val annotations -> annotations/captions_val2017.json

Example:
  python evaluation/prepare_coco_subset.py \
      --annotations ~/datasets/coco/annotations/captions_val2017.json \
      --images ~/datasets/coco/val2017 \
      --out evaluation/data/coco --num-images 500
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from app.services.evaluation.datasets import prepare_coco_subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("evaluation/data/coco"))
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    count = prepare_coco_subset(
        annotations_path=args.annotations.expanduser(),
        images_dir=args.images.expanduser(),
        output_dir=args.out,
        num_images=args.num_images,
        seed=args.seed,
    )
    print(f"Wrote {count} images and queries to {args.out}")


if __name__ == "__main__":
    main()
