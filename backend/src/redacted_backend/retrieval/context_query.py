from __future__ import annotations

import re
import unicodedata

from redacted_backend.scientific_config import ScientificConfig

_TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", flags=re.UNICODE)


class ContextQueryError(ValueError):
    pass


def build_context_query(
    paragraph_window: str,
    scientific: ScientificConfig | None = None,
) -> str:
    config = scientific or ScientificConfig()
    normalized = unicodedata.normalize("NFKC", paragraph_window).casefold()
    stopwords = frozenset(config.context_stopwords)
    tokens = [token for token in _TOKEN.findall(normalized) if token not in stopwords]
    if not tokens:
        raise ContextQueryError("Context window contains no searchable visible terms")
    return " ".join(tokens)
