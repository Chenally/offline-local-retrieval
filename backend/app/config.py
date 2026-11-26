from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.services.retrieval.ranking import FusionMethod, RankingConfig, SearchMode

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Filesystem locations and tunables for the local service.

    Every value can be overridden with an ``OLR_*`` environment variable,
    which is how tests and packaged builds point the service elsewhere.
    """

    runtime_dir: Path
    models_dir: Path
    ranking: RankingConfig
    chunk_max_units: int = 80
    chunk_overlap_units: int = 20
    candidate_multiplier: int = 5

    @classmethod
    def from_env(cls) -> AppSettings:
        runtime_dir = (
            Path(os.environ.get("OLR_RUNTIME_DIR", PROJECT_ROOT / "runtime"))
            .expanduser()
            .resolve()
        )
        models_dir = (
            Path(os.environ.get("OLR_MODELS_DIR", PROJECT_ROOT / "models"))
            .expanduser()
            .resolve()
        )

        ranking = RankingConfig(
            weights={
                SearchMode.KEYWORD: float(os.environ.get("OLR_WEIGHT_KEYWORD", 0.3)),
                SearchMode.TEXT_SEMANTIC: float(
                    os.environ.get("OLR_WEIGHT_TEXT_SEMANTIC", 0.7)
                ),
                SearchMode.IMAGE_SEMANTIC: float(
                    os.environ.get("OLR_WEIGHT_IMAGE_SEMANTIC", 0.7)
                ),
            },
            method=FusionMethod(os.environ.get("OLR_FUSION_METHOD", "calibrated")),
        )

        return cls(
            runtime_dir=runtime_dir,
            models_dir=models_dir,
            ranking=ranking,
        )

    @property
    def chroma_dir(self) -> Path:
        return self.runtime_dir / "chroma"

    @property
    def keyword_db_path(self) -> Path:
        return self.runtime_dir / "index" / "keyword.sqlite3"

    @property
    def manifest_db_path(self) -> Path:
        return self.runtime_dir / "index" / "manifest.sqlite3"

    @property
    def text_model_path(self) -> Path:
        """Sentence embedder for text-to-text search (all-MiniLM-L6-v2)."""

        return self.models_dir / "minilm_l6_text.tflite"

    @property
    def text_vocab_path(self) -> Path:
        return self.models_dir / "minilm_l6_vocab.txt"

    @property
    def bert_model_path(self) -> Path:
        """Week 3 MediaPipe bert_embedder; kept only for diagnostics."""

        return self.models_dir / "bert_text_embedder.tflite"

    @property
    def bert_vocab_path(self) -> Path:
        return self.models_dir / "bert_vocab.txt"

    @property
    def mobileclip_text_model_path(self) -> Path:
        return self.models_dir / "mobileclip_s0_text.tflite"

    @property
    def mobileclip_image_model_path(self) -> Path:
        return self.models_dir / "mobileclip_s0_image.tflite"

    @property
    def mobileclip_tokenizer_path(self) -> Path:
        return self.models_dir / "mobileclip_s0_tokenizer.json"
