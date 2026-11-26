"""Check whether the local text embedders capture sentence meaning.

Two modes:

1. Project mode (inside the project's virtual environment):

       python tools/diagnose_text_embeddings.py

   Embeds sentence pairs with the text-semantic backend and the MobileCLIP text
   encoder, prints related vs unrelated cosine similarities, inspects the
   BERT vocabulary/tokenization, and saves the BERT vectors to
   runtime/diagnostics/bert_vectors.npy.

2. MediaPipe reference mode (in a *separate* environment that only has
   ``mediapipe`` and ``numpy`` installed, so it cannot disturb project
   dependencies):

       python tools/diagnose_text_embeddings.py --mediapipe

   Embeds the same sentences with MediaPipe's own TextEmbedder and the
   Week 3 bert_embedder model file (models/bert_text_embedder.tflite).
   If MediaPipe separates the topics but the project backend
   does not, the bug is in our preprocessing; if both fail, the model
   itself is unsuitable for sentence similarity.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "runtime" / "diagnostics"
OUR_VECTORS = OUTPUT_DIR / "bert_vectors.npy"

# (anchor, related, unrelated)
TRIPLES = [
    (
        "how do I make tomato soup",
        "Tomato soup recipe: cook tomatoes with garlic and basil, then blend.",
        "This project searches local documents without sending data to the cloud.",
    ),
    (
        "The cat sat on the mat.",
        "A kitten is resting on a rug.",
        "Quarterly revenue grew by ten percent.",
    ),
    (
        "Where are my tax documents?",
        "Locating last year's tax return files.",
        "The football match ended in a draw.",
    ),
    (
        "searching private files offline",
        "Local search that keeps documents on the device.",
        "Slow cooked beef stew with carrots.",
    ),
    (
        "machine learning model inference on a phone",
        "Running neural networks on mobile devices.",
        "Gardening tips for growing roses in spring.",
    ),
]


def sentences() -> list[str]:
    return [text for triple in TRIPLES for text in triple]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def report(name: str, vectors: np.ndarray) -> float:
    print(f"\n=== {name} ===")
    print(f"{'related':>8} {'unrelated':>10} {'margin':>8}  anchor")
    margins: list[float] = []
    correct = 0

    for index, (anchor, _, _) in enumerate(TRIPLES):
        a, r, u = vectors[3 * index : 3 * index + 3]
        related = cosine(a, r)
        unrelated = cosine(a, u)
        margins.append(related - unrelated)
        correct += related > unrelated
        print(f"{related:8.3f} {unrelated:10.3f} {related - unrelated:8.3f}  {anchor}")

    all_pairs = [
        cosine(vectors[i], vectors[j])
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]
    mean_margin = float(np.mean(margins))
    print(
        f"ordering correct: {correct}/{len(TRIPLES)}, "
        f"mean margin {mean_margin:.3f}, "
        f"all-pairs cosine range [{min(all_pairs):.3f}, {max(all_pairs):.3f}]"
    )
    return mean_margin


def project_mode() -> int:
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))

    from tokenizers import BertWordPieceTokenizer

    from app.config import AppSettings
    from app.container import load_backends
    from app.services.embeddings.service import EmbeddingService
    from app.services.embeddings.types import EmbeddingSpace

    settings = AppSettings.from_env()

    vocab = settings.text_vocab_path.read_text(encoding="utf-8").splitlines()
    uppercase = sum(1 for token in vocab if any(ch.isupper() for ch in token))
    print(f"Text vocabulary: {len(vocab)} tokens, {uppercase} contain uppercase")
    print(f"First tokens: {vocab[:5]}  [CLS]={vocab.index('[CLS]') if '[CLS]' in vocab else None}")

    tokenizer = BertWordPieceTokenizer(str(settings.text_vocab_path), lowercase=True)
    encoding = tokenizer.encode("How do I make Tomato Soup?")
    print(f"Tokens: {encoding.tokens}")
    print(f"IDs:    {encoding.ids}")

    backends, status = load_backends(settings)
    print(f"Backends: {status}")
    service = EmbeddingService(backends)
    texts = sentences()

    bert = np.asarray(
        [v.values for v in service.embed_texts(texts, space=EmbeddingSpace.TEXT_SEMANTIC)],
        dtype=np.float32,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUR_VECTORS, bert)
    report("Text semantic backend (project)", bert)

    if status.get("mobileclip") == "ready":
        clip = np.asarray(
            [v.values for v in service.embed_texts(texts, space=EmbeddingSpace.MULTIMODAL)],
            dtype=np.float32,
        )
        report("MobileCLIP text encoder (for comparison)", clip)

    print(f"\nSaved BERT vectors to {OUR_VECTORS}")
    return 0


def mediapipe_mode() -> int:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import text

    options = text.TextEmbedderOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(MODELS_DIR / "bert_text_embedder.tflite")
        ),
        l2_normalize=True,
    )

    with text.TextEmbedder.create_from_options(options) as embedder:
        reference = np.asarray(
            [embedder.embed(sentence).embeddings[0].embedding for sentence in sentences()],
            dtype=np.float32,
        )

    report("BERT via MediaPipe TextEmbedder (reference)", reference)

    if OUR_VECTORS.is_file() and np.load(OUR_VECTORS).shape == reference.shape:
        ours = np.load(OUR_VECTORS)
        agreement = [cosine(a, b) for a, b in zip(ours, reference, strict=True)]
        print(
            "\nProject vs MediaPipe vectors, per-sentence cosine: "
            f"min {min(agreement):.4f}, mean {np.mean(agreement):.4f}"
        )
        print(
            "~1.0 means our preprocessing matches MediaPipe; "
            "clearly lower means a tokenization or input-mapping bug."
        )
    else:
        print("\nRun project mode first to compare against the project backend.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mediapipe", action="store_true")
    args = parser.parse_args()
    return mediapipe_mode() if args.mediapipe else project_mode()


if __name__ == "__main__":
    sys.exit(main())
