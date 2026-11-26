"""Check that all five model assets are present, match Week 3, and load.

Usage (from the repository root, inside the virtual environment):
    python tools/verify_models.py

Works on Windows, macOS and Linux (no ``shasum`` needed).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import AppSettings  # noqa: E402
from app.container import load_backends  # noqa: E402
from app.services.embeddings.service import EmbeddingService  # noqa: E402
from app.services.embeddings.types import EmbeddingSpace  # noqa: E402

# Model files required at runtime. Values are the recorded SHA-256 of the
# validated build, or None when no reference build has been recorded yet.
REQUIRED_ASSETS: dict[str, str | None] = {
    "minilm_l6_text.tflite": "6c806391d28bed212fff555a6cc6583bc0caff2d213ca3ad10e6f523733e2c4d",
    "minilm_l6_vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    "mobileclip_s0_text.tflite": (
        "5aaff7d1414855d26fcf3744503d4dba238336ebaaecd8282d0358235ba6262c"
    ),
    "mobileclip_s0_image.tflite": (
        "429d16fca1f2adaf96f3a43744af88b1cc88a6202617f1a2ca2affb91dbf2c7c"
    ),
    "mobileclip_s0_tokenizer.json": (
        "6d9109cc838977f3ca94a379eec36aecc7c807e1785cd729660ca2fc0171fb35"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> int:
    settings = AppSettings.from_env()
    print(f"Models directory: {settings.models_dir}\n")
    missing = False

    for name, expected in REQUIRED_ASSETS.items():
        path = settings.models_dir / name

        if not path.is_file():
            print(f"  MISSING  {name}")
            missing = True
            continue

        actual = sha256(path)
        size_mb = path.stat().st_size / (1024 * 1024)
        if expected is None:
            status = "no recorded checksum"
        elif actual == expected:
            status = "matches recorded build"
        else:
            status = "different build"
        print(f"  {name:<30} {size_mb:7.1f} MB  {actual}  ({status})")

    if missing:
        print(
            "\nSome files are missing. Run tools/convert_minilm_l6.ipynb (text) and "
            "tools/convert_mobileclip_s0.ipynb (MobileCLIP) in Google Colab."
        )
        return 1

    print("\nLoading backends...")
    backends, status = load_backends(settings)

    for component, state in status.items():
        print(f"  {component}: {state}")

    if any(state != "ready" for state in status.values()):
        return 1

    service = EmbeddingService(backends)
    sample = Path(PROJECT_ROOT / "sample_data" / "images" / "sample.jpg")

    text = service.embed_texts(
        ["offline local search"], space=EmbeddingSpace.TEXT_SEMANTIC
    )
    clip_text = service.embed_texts(["a photo"], space=EmbeddingSpace.MULTIMODAL)
    image = service.embed_images([sample]) if sample.is_file() else []

    print("\nSmoke inference:")
    print(f"  Sentence text vector:    dim={len(text[0].values)}")
    print(f"  MobileCLIP text vector:  dim={len(clip_text[0].values)}")

    if image:
        similarity = sum(
            a * b for a, b in zip(image[0].values, clip_text[0].values, strict=True)
        )
        print(f"  MobileCLIP image vector: dim={len(image[0].values)}")
        print(f"  cos('a photo', sample.jpg) = {similarity:.4f}")

    print("\nAll models loaded and produced embeddings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
