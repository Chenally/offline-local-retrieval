from pathlib import Path

import pypdfium2 as pdfium


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "sample_data" / "pdf" / "sample.pdf"


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Test PDF was not found: {PDF_PATH}"
        )

    document = pdfium.PdfDocument(str(PDF_PATH))

    if len(document) == 0:
        raise RuntimeError("The PDF does not contain any pages.")

    extracted_pages: list[str] = []

    # Week 1 only needs a small validation, so reading the first
    # three pages is sufficient.
    page_count = min(len(document), 3)

    for page_index in range(page_count):
        page = document[page_index]
        text_page = page.get_textpage()
        text = text_page.get_text_range()

        extracted_pages.append(text.strip())

        text_page.close()
        page.close()

    document.close()

    extracted_text = "\n".join(extracted_pages).strip()

    if not extracted_text:
        raise RuntimeError(
            "PDFium opened the PDF but did not extract any text."
        )

    print(f"PDF path: {PDF_PATH}")
    print(f"Pages checked: {page_count}")
    print("Extracted text:")
    print(extracted_text[:500])
    print("PDFium parsing check passed.")


if __name__ == "__main__":
    main()
