from __future__ import annotations

import hashlib
import re
import unicodedata

from redacted_backend.domain.visible_examples import VisibleExample
from redacted_backend.retrieval.types import PreparedContextQuery
from redacted_backend.scientific_config import ScientificConfig

_TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", flags=re.UNICODE)
_TARGET_MARKER = re.compile(r"\[\[TARGET\]\]", flags=re.IGNORECASE)
_STRUCTURAL_MARKER = re.compile(r"\[\[TARGET\]\]|\[REDACTED\]", flags=re.IGNORECASE)


class ContextQueryError(ValueError):
    pass


def build_context_query(
    paragraph_window: str,
    scientific: ScientificConfig | None = None,
) -> str:
    config = scientific or ScientificConfig()
    visible_text = _STRUCTURAL_MARKER.sub(" ", paragraph_window)
    normalized = unicodedata.normalize("NFKC", visible_text).casefold()
    stopwords = frozenset(config.context_stopwords)
    tokens = [token for token in _TOKEN.findall(normalized) if token not in stopwords]
    if not tokens:
        raise ContextQueryError("Context window contains no searchable visible terms")
    return " ".join(tokens)


def prepare_context_query(
    example: VisibleExample,
    scientific: ScientificConfig | None = None,
) -> PreparedContextQuery:
    validate_context_query_example(example)
    config = scientific or ScientificConfig()
    text = build_context_query(example.context_window, config)
    return PreparedContextQuery(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        term_count=len(text.split()),
        stopwords_version=config.context_stopwords_version,
    )


def validate_context_query_example(example: VisibleExample) -> None:
    if len(_TARGET_MARKER.findall(example.context_window)) != 1:
        raise ContextQueryError("Context window must contain exactly one target marker")
