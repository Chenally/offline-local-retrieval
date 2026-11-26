# Offline Accessible Multimodal Local Content Retrieval System

An offline-first, cross-platform application for discovering and retrieving information from local documents and images. The project is designed to make personal content easier to search without uploading private files to cloud services, while targeting WCAG 2.1 AA accessibility requirements.

> **Development status:** This is an active eight-week software engineering project. Weeks 1 through 4 are complete; the Week 5 Flutter UI and accessibility work is the next milestone.

## Project Motivation

Local devices often contain information spread across text files, PDFs, office documents, screenshots, and images. Traditional search tools rely heavily on filenames or exact keyword matches, while many semantic-search products require cloud processing.

This project aims to provide:

- Offline processing and storage for local content
- Semantic retrieval across text and images
- A cross-platform user interface
- Accessible keyboard and screen-reader workflows
- Modular, testable, and maintainable components
- Transparent and explainable ranking behavior

## Current Progress

### Week 1 - Completed

- Established the Python 3.11 virtual environment and repository structure
- Created a FastAPI service with a liveness endpoint
- Validated persistent ChromaDB storage and vector queries
- Validated TXT and PDF extraction with Apache Tika and PDFium
- Validated local LiteRT/TensorFlow Lite execution
- Created and built the initial Flutter macOS application
- Added sample local files, test utilities, and Git ignore rules

### Week 2 - Completed

- Defined stable interfaces for file scanning, parsing, and ingestion
- Implemented recursive local file discovery
- Added file metadata collection and SHA-256 duplicate detection
- Implemented TXT and PDF parsers
- Added recoverable batch-ingestion error handling
- Added tests for valid, corrupted, unsupported, duplicate, and permission-restricted files
- Completed the Week 2 file-ingestion foundation and parsing regression suite

### Week 3 - Completed

- Added separate `TEXT_SEMANTIC` and `MULTIMODAL` embedding spaces
- Integrated a local BERT-compatible LiteRT backend for text-to-text retrieval
- Converted and integrated MobileCLIP-S0 text and image LiteRT encoders
- Added BERT WordPiece and MobileCLIP tokenizers
- Added text and image preprocessing with model tensor-contract validation
- Added a unified embedding service with L2 normalization and in-memory caching
- Verified MobileCLIP LiteRT outputs against the official PyTorch checkpoint
- Verified real BERT and MobileCLIP inference on macOS
- Completed 77 backend tests with 96.19% embedding-package coverage
- Documented the implementation in [`docs/week3/README.md`](docs/week3/README.md)

### Week 4 - Completed

- Added DOCX parsing (`python-docx`) and JPG/PNG image ingestion
- Added page-aware, overlapping text chunking sized for the BERT input
- Integrated persistent ChromaDB collections bound to one embedding space and model each
- Added a SQLite FTS5 keyword index with BM25 scoring and Chinese phrase support
- Added a manifest for incremental indexing of new, changed, deleted and duplicate files
- Implemented explainable hybrid ranking (weighted min-max or reciprocal rank fusion)
- Added extension, folder and date filters
- Exposed indexing jobs, index status, readiness and search through FastAPI
- Added NQ and COCO benchmark preparation and a retrieval benchmark runner
- Replaced the Week 3 MediaPipe BERT embedder with all-MiniLM-L6-v2 after diagnostics showed it could not separate related from unrelated sentences
- Documented the implementation in [`docs/week4/README.md`](docs/week4/README.md)

## Planned Features

- Local folder selection and recursive file discovery
- Parsing for TXT, PDF, DOCX, JPG, and PNG files
- SHA-256-based duplicate detection and incremental indexing
- Offline text embeddings using all-MiniLM-L6-v2 (BERT architecture) on LiteRT
- Offline text-to-image retrieval using MobileCLIP-compatible encoders
- Local vector storage and similarity search with ChromaDB
- Explainable hybrid ranking using keyword, text-semantic, and image-semantic scores
- Search filters, indexing status, and recoverable error messages
- Keyboard-only navigation, screen-reader labels, scalable text, and high-contrast presentation
- Production builds for supported desktop platforms

## System Architecture

```mermaid
flowchart LR
    A[Local TXT, PDF, DOCX, JPG and PNG files]
    B[File Scanner]
    C[Metadata and SHA-256 Deduplication]
    D[Parser Registry]
    E[Normalized Documents]
    F[Offline Embedding Engine]
    G[ChromaDB Vector Storage]
    H[Hybrid Retrieval and Ranking]
    I[Local FastAPI Service]
    J[Accessible Flutter UI]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
```

The architecture separates file input/output, parsing, embedding, vector storage, retrieval, and presentation. Each parser returns a normalized document model so later embedding and retrieval modules do not depend on individual file formats.

## Technology Stack

| Area | Technology |
| --- | --- |
| Backend service | Python 3.11, FastAPI, Uvicorn |
| Cross-platform UI | Flutter, Dart |
| Document parsing | PDFium through `pypdfium2`, `python-docx` |
| Local inference | LiteRT / TensorFlow Lite |
| Text model | all-MiniLM-L6-v2 (BERT architecture) on LiteRT, 384 dimensions |
| Multimodal model | MobileCLIP-S0 text and image LiteRT encoders |
| Vector storage | ChromaDB, SQLite FTS5 (keyword BM25) |
| Backend testing | pytest, pytest-cov |
| UI testing | Flutter Test |
| Version control | Git, GitHub |
| Accessibility target | WCAG 2.1 AA |

## File-Format Support

| Format | Status | Intended processing |
| --- | --- | --- |
| TXT | Supported | UTF-8 text extraction, chunking, BERT embedding |
| PDF | Supported | PDFium per-page text extraction, chunking, BERT embedding |
| DOCX | Supported | `python-docx` paragraph and table extraction |
| JPG | Supported | Image validation and MobileCLIP embedding |
| PNG | Supported | Image validation and MobileCLIP embedding |

## Repository Structure

```text
offline_local_retrieval/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── services/
│   ├── tests/
│   └── requirements.lock
├── frontend/
│   ├── lib/
│   ├── macos/
│   └── test/
├── docs/
├── evaluation/
├── models/
├── runtime/
├── sample_data/
├── tools/
├── .gitignore
└── README.md
```

`models/` and `runtime/` contain local model and runtime data and are intentionally excluded from Git where appropriate.

## Development Setup

### Prerequisites

- Python 3.11
- Git
- Flutter SDK
- Xcode for macOS desktop development

The current development environment has been validated on Apple Silicon macOS. Additional desktop platforms will be validated during the packaging phase.

### 1. Clone the repository

```bash
git clone <repository-url>
cd offline-local-retrieval
```

### 2. Create the Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.lock
```

### 3. Run the backend service

```bash
python -m uvicorn app.main:app \
  --app-dir backend \
  --host 127.0.0.1 \
  --port 8765 \
  --reload
```

Verify the service from another terminal:

```bash
curl http://127.0.0.1:8765/health/live
```

Expected response:

```json
{"status":"ok"}
```

### 4. Run the Flutter application

```bash
cd frontend
flutter pub get
flutter run -d macos
```

## Testing

Run the backend test suite from the repository root:

```bash
PYTHONPATH=backend python -m pytest backend/tests -v
```

Run the coverage check:

```bash
PYTHONPATH=backend python -m pytest backend/tests -v \
  --cov=app.services \
  --cov-report=term-missing \
  --cov-fail-under=85
```

Search the local index once the service is running:

```bash
curl -s -X POST http://127.0.0.1:8765/index \
  -H 'Content-Type: application/json' \
  -d "{\"directory\": \"$PWD/sample_data\"}"

curl -s -X POST http://127.0.0.1:8765/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "local documents without the cloud", "top_k": 5}'
```

See [`docs/week4/README.md`](docs/week4/README.md) for the full API and test plan.

Run Flutter tests:

```bash
cd frontend
flutter test
```

## Privacy and Accessibility Principles

### Privacy

- User-selected files remain on the local device
- Parsing, embedding, indexing, and retrieval are designed to run offline
- Local vectors and metadata are stored in the local runtime directory
- No cloud upload is required for the main workflow

### Accessibility

The final interface will target WCAG 2.1 AA and include:

- Semantic labels for interactive controls
- Keyboard-only navigation and visible focus states
- VoiceOver and NVDA-compatible announcements
- Dynamic text scaling
- High-contrast colors
- Sufficiently large interactive controls
- Clear and recoverable error messages

## Eight-Week Roadmap

| Week | Focus | Status |
| --- | --- | --- |
| 1 | Requirements, repository setup, and technical validation | Completed |
| 2 | Architecture and file-ingestion foundation | Completed |
| 3 | Offline text and multimodal embedding engine | Completed |
| 4 | ChromaDB integration, hybrid ranking, and retrieval MVP | Completed |
| 5 | Flutter UI and accessibility implementation | Planned |
| 6 | Integration, testing, privacy review, and performance optimization | Planned |
| 7 | Documentation, open-source compliance, and release preparation | Planned |
| 8 | Packaging, final validation, GitHub release, and portfolio delivery | Planned |

## Known Limitations

- The project is not yet production-ready
- Only the macOS development environment has been validated so far
- The Flutter interface is not yet connected to the retrieval API
- Scanned documents without a text layer are searchable by file name only; OCR is out of scope
- Accessibility validation and clean-environment offline testing are scheduled for later milestones

## License

A project license has not yet been finalized. Apache License 2.0 is planned after the Week 7 dependency and open-source compliance review.

## Author

Junhan Chen
