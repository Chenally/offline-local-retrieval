"""Download the BERT text embedder and extract its WordPiece vocabulary.

Legacy: the service now uses all-MiniLM-L6-v2 (tools/convert_minilm_l6.ipynb).
This Week 3 model is only needed to rerun tools/diagnose_text_embeddings.py
in --mediapipe mode.

The model is MediaPipe's ``bert_embedder`` (float32, version 1), which
matches the backend's model ID ``bert_embedder_float32_v1``. Its
vocabulary is packed inside the .tflite file as TFLite metadata, so no
separate download is needed.

Usage (from the repository root):
    python tools/fetch_bert_model.py
    python tools/fetch_bert_model.py --model path/to/bert_embedder.tflite  # offline
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/text_embedder/"
    "bert_embedder/float32/1/bert_embedder.tflite"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "bert_text_embedder.tflite"
VOCAB_PATH = MODELS_DIR / "bert_vocab.txt"

# Checksums recorded during Week 3 validation (docs/week3/README.md).
EXPECTED_SHA256 = {
    MODEL_PATH.name: "02ae6279faf86c2cd4ff18f61876c878bcc0b572b472f0678897a184c4ac7ef6",
    VOCAB_PATH.name: "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}")

    with urllib.request.urlopen(url, timeout=60) as response:
        with temporary.open("wb") as file:
            shutil.copyfileobj(response, file)

    temporary.replace(destination)


def extract_vocab(model_path: Path, destination: Path) -> str:
    """Extract the vocabulary that TFLite metadata stores as a zip archive."""

    try:
        archive = zipfile.ZipFile(model_path)
    except zipfile.BadZipFile as exc:
        raise SystemExit(
            f"{model_path} has no packed metadata files; cannot find vocabulary"
        ) from exc

    with archive:
        names = archive.namelist()
        candidates = [name for name in names if name.lower().endswith(".txt")]

        if not candidates:
            raise SystemExit(f"No vocabulary file found in metadata: {names}")

        name = next(
            (item for item in candidates if "vocab" in item.lower()), candidates[0]
        )
        destination.write_bytes(archive.read(name))

    return name


def check(path: Path) -> bool:
    actual = sha256(path)
    expected = EXPECTED_SHA256[path.name]
    status = "OK" if actual == expected else "DIFFERENT from Week 3"
    print(f"  {path.name}: {actual}  [{status}]")
    return actual == expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        type=Path,
        help="Use an already downloaded bert_embedder.tflite instead of downloading",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.model:
        shutil.copy2(args.model.expanduser(), MODEL_PATH)
    else:
        download(MODEL_URL, MODEL_PATH)

    vocab_name = extract_vocab(MODEL_PATH, VOCAB_PATH)
    print(f"Extracted '{vocab_name}' -> {VOCAB_PATH}")
    print("SHA-256:")
    matches = [check(MODEL_PATH), check(VOCAB_PATH)]

    if not all(matches):
        print(
            "Checksums differ from Week 3. The upstream file may have changed; "
            "run tools/verify_models.py to confirm the tensor contract still holds."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
