from __future__ import annotations

import hashlib
import re
import unicodedata

from pydantic import BaseModel, ConfigDict, Field

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


class CorpusDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str = Field(min_length=1)
    classification: str | None
    authored: str | None
    title: str | None
    body: str
    source: str = Field(min_length=1)
    pg_cnt: int | None = Field(default=None, ge=0)
    char_cnt: int = Field(ge=0)
    word_cnt: int = Field(ge=0)


class CorpusChunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_id: str
    ordinal: int = Field(ge=0)
    text: str
    word_count: int = Field(ge=1, le=300)
    title: str | None
    source: str
    classification: str | None
    authored: str | None


def chunk_document(
    document: CorpusDocument,
    max_words: int = 300,
) -> list[CorpusChunk]:
    if max_words < 1:
        raise ValueError("max_words must be positive")
    paragraphs = _normalize_paragraphs(document.body)
    units: list[tuple[int, str]] = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        units.extend(
            (paragraph_index, unit) for unit in _split_paragraph(paragraph, max_words)
        )

    texts: list[str] = []
    current = ""
    current_words = 0
    current_paragraph: int | None = None
    for paragraph_index, unit in units:
        unit_words = len(unit.split())
        separator = ""
        if current:
            separator = "\n\n" if current_paragraph != paragraph_index else " "
        if current and current_words + unit_words > max_words:
            texts.append(current)
            current = unit
            current_words = unit_words
        else:
            current = f"{current}{separator}{unit}"
            current_words += unit_words
        current_paragraph = paragraph_index
    if current:
        texts.append(current)

    return [
        CorpusChunk(
            id=_chunk_id(document.doc_id, ordinal, text),
            document_id=document.doc_id,
            ordinal=ordinal,
            text=text,
            word_count=len(text.split()),
            title=document.title,
            source=document.source,
            classification=document.classification,
            authored=document.authored,
        )
        for ordinal, text in enumerate(texts)
    ]


def _normalize_paragraphs(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n")
    return [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in _PARAGRAPH_BREAK.split(normalized)
        if paragraph.strip()
    ]


def _split_paragraph(paragraph: str, max_words: int) -> list[str]:
    if len(paragraph.split()) <= max_words:
        return [paragraph]
    sentences = [
        item.strip() for item in _SENTENCE_BREAK.split(paragraph) if item.strip()
    ]
    units: list[str] = []
    current: list[str] = []
    current_count = 0
    for sentence in sentences:
        sentence_words = sentence.split()
        if len(sentence_words) > max_words:
            if current:
                units.append(" ".join(current))
                current = []
                current_count = 0
            units.extend(
                " ".join(sentence_words[index : index + max_words])
                for index in range(0, len(sentence_words), max_words)
            )
            continue
        if current and current_count + len(sentence_words) > max_words:
            units.append(" ".join(current))
            current = []
            current_count = 0
        current.extend(sentence_words)
        current_count += len(sentence_words)
    if current:
        units.append(" ".join(current))
    return units


def _chunk_id(document_id: str, ordinal: int, text: str) -> str:
    value = f"{document_id}\0{ordinal}\0{text}".encode()
    return hashlib.sha256(value).hexdigest()
