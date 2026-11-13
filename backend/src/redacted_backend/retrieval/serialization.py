from __future__ import annotations

from redacted_backend.domain.retrieval import FusedDocument


def serialize_prompt_documents(documents: list[FusedDocument]) -> str:
    rendered: list[str] = []
    for document_number, document in enumerate(documents, start=1):
        lines = [f"[DOCUMENT {document_number}]"]
        for excerpt_number, chunk in enumerate(document.chunks, start=1):
            lines.extend((f"[EXCERPT {excerpt_number}]", chunk.text.strip()))
        rendered.append("\n".join(lines))
    return "\n\n".join(rendered)
