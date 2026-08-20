# Week 1: Scope, Requirements and Technical Validation

## 1. Project Scope

The project will implement an offline-first, cross-platform local content
retrieval application. The system will parse local documents and images,
generate embeddings through local LiteRT models, store vectors in ChromaDB,
and provide an accessible Flutter interface.

The initial implementation will prioritize macOS, TXT and PDF. DOCX, JPG,
PNG, Windows and Linux support will be added after the core workflow is
stable.

Cloud storage, user accounts, cloud inference and mobile deployment are
outside the initial eight-week scope.

## 2. Functional Requirements

- Import supported local documents and images.
- Extract text and metadata from local files.
- Generate text and image embeddings locally.
- Store and retrieve vectors using ChromaDB.
- Search local content using semantic similarity.
- Display ranked results in a Flutter interface.
- Keep local content on the user's device.

## 3. Non-Functional Requirements

- The core retrieval workflow must operate without an internet connection.
- Local files and extracted content must not be uploaded externally.
- The interface will target WCAG 2.1 AA accessibility requirements.
- The codebase will use modular components and automated tests.
- macOS will be the first fully validated platform.

## 4. Technical Assumptions

- Python 3.11 is used for backend development.
- Flutter is used for the cross-platform interface.
- Apache Tika and PDFium are used for document parsing.
- LiteRT is used for offline model inference.
- ChromaDB is used for local vector storage.
- Initial development is performed on Apple Silicon macOS.
- Large public datasets will be limited to small validation subsets.

## 5. Repository Structure

- `backend/`: FastAPI service and Python tests.
- `frontend/`: Flutter desktop application.
- `tools/`: Technical validation scripts.
- `models/`: Local LiteRT models excluded from Git.
- `sample_data/`: Small local validation files.
- `runtime/`: Local ChromaDB data excluded from Git.
- `evaluation/`: Future evaluation scripts and results.
- `docs/`: Project documentation.

## 6. Coding Standards

Python code will use type hints, descriptive names and small functions.
Pytest will be used for Python testing, and Ruff will be used for linting.
Flutter code will follow Dart formatting and analysis rules. Runtime
databases, virtual environments, generated build files and local model
binaries will not be committed to Git.

## 7. Accessibility and Open-Source Requirements

The Flutter interface will support semantic labels, keyboard navigation,
scalable text, sufficient color contrast and screen readers. Dependencies
will use compatible open-source licenses, and their licenses will be reviewed
before the final release.

## 8. Environment Validation

| Component | Version | Result |
|---|---|---|
| Python | 3.11.16 | Passed |
| Git | 2.50.1 | Passed |
| FastAPI | 0.141.1 | API and test passed |
| ChromaDB | 1.5.9 | Insert and query test passed |
| Apache Tika | 3.3.2 | TXT and PDF parsing passed |
| PDFium | 5.13.0| PDF text extraction passed |
| LiteRT | 2.2.0 | Model inference passed |
| Flutter | 3.47.1 | macOS build passed |
| Dart | 3.13.1 | Passed |
| Xcode | 26.6 | macOS toolchain passed |
| CocoaPods | 1.17.0 | Passed |

## 9. Validation Dataset

The Week 1 repository contains self-created TXT, PDF and JPG samples. These
files are used for technical smoke tests. Large datasets such as Natural
Questions, COCO, RVL-CDIP and Wikipedia will be restricted to small,
representative validation subsets during later development.

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| BERT or MobileCLIP is too slow on CPU | High | Use smaller compatible variants |
| Model conversion fails | High | Validate models early and maintain fallback models |
| Dataset size is too large | Medium | Use small representative validation subsets |
| File parsing differs across formats | Medium | Stabilize TXT and PDF before adding more formats |
| Cross-platform behavior differs | Medium | Complete macOS first and validate other platforms later |
| Accessibility work is delayed | High | Include accessibility requirements during UI development |

## 11. Week 1 Result

The development environment and repository structure have been created.
FastAPI, ChromaDB, Apache Tika, PDFium, LiteRT and Flutter were validated
through small local experiments.

Status: Ready for supervisor review.
