from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from redacted_backend.domain.retrieval import RetrievalTrace
from redacted_backend.domain.visible_examples import VisibleExample

_PROMPT_ROOT = Path(__file__).resolve().parents[3] / "prompts"
_QUERY_PATH = _PROMPT_ROOT / "realistic_query_v1.txt"
_PREDICTION_PATH = _PROMPT_ROOT / "realistic_prediction_v1.txt"


class RenderedPrompt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_version: str


def render_query_prompt(example: VisibleExample) -> RenderedPrompt:
    return _render(
        _QUERY_PATH,
        "realistic-query-v1",
        date=_paper_date(example),
        bucket=example.length_cue.bucket,
        before=example.before_target,
        after=example.after_target,
    )


def render_prediction_prompt(
    example: VisibleExample,
    retrieval: RetrievalTrace,
) -> RenderedPrompt:
    return _render(
        _PREDICTION_PATH,
        "realistic-prediction-v1",
        date=_paper_date(example),
        hint=f"It is {example.length_cue.bucket} long. ",
        documents=retrieval.serialized_documents,
        before=example.before_target,
        after=example.after_target,
    )


def _render(path: Path, version: str, **values: str) -> RenderedPrompt:
    template = path.read_text(encoding="utf-8")
    text = template.format(**values).rstrip("\n")
    return RenderedPrompt(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        template_sha256=hashlib.sha256(template.encode("utf-8")).hexdigest(),
        template_version=version,
    )


def _paper_date(example: VisibleExample) -> str:
    value = example.document_date
    return f"{value.strftime('%B')} {value.day}, {value.year}"
