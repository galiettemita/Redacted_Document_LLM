# Generated public PDF fixtures

These fixtures contain only invented text. Run `generate_fixtures.py` from this directory to regenerate them with PyMuPDF.

- `visible-layout.pdf` has a visibly printed date, two text columns, a repeated header, a black target box, a wide ordinary whitespace gap, and embedded text painted beneath the black box. The whitespace gap is a negative control: it must not become a persisted passage without independent redaction evidence.
- `scanned-visible-layout.pdf` is image-only and exercises the real Tesseract fallback.

Neither fixture contains an answer, private document, or material copied from Andrew Tang's corpus.
