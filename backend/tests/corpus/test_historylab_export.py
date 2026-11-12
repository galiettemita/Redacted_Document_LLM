from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from redacted_backend.corpus.historylab_export import (
    CAPTURE_SELECT_FIELDS,
    ApiCaptureArtifact,
    ApiCaptureManifest,
    CorpusCaptureError,
    capture_historylab_corpus,
)


def api_document(doc_id: str, corpus: str = "cfpf") -> dict[str, object]:
    body = f"Visible body for {doc_id}."
    return {
        "doc_id": doc_id,
        "corpus": corpus,
        "classification": "unclassified",
        "authored": "1975-01-02T00:00:00+00:00",
        "title": f"Title {doc_id}",
        "body": body,
        "source": f"https://example.test/{doc_id}",
        "pg_cnt": 1,
        "char_cnt": len(body),
        "word_cnt": len(body.split()),
        "doc_lang": "en",
    }


def response_for(
    request: httpx.Request,
    *,
    rows: list[dict[str, object]],
    corpus: str = "cfpf",
    count: int | Callable[[int], int] | None = None,
    count_call: list[int] | None = None,
) -> httpx.Response:
    if request.url.path == "/corpora":
        catalog_count = count(0) if callable(count) else count
        return httpx.Response(
            200,
            json=[
                {
                    "corpus": corpus,
                    "doc_cnt": catalog_count
                    if catalog_count is not None
                    else len(rows),
                    "title": "Test",
                }
            ],
        )
    params = request.url.params
    if request.url.path != "/docs":
        return httpx.Response(404)
    if request.headers.get("Prefer") == "count=exact":
        index = 0 if count_call is None else count_call[0]
        if count_call is not None:
            count_call[0] += 1
        exact = (
            count(index)
            if callable(count)
            else (count if count is not None else len(rows))
        )
        return httpx.Response(
            200,
            json=rows[:1],
            headers={"Content-Range": f"0-0/{exact}"},
        )
    last_id_filter = params.get("doc_id")
    last_id = last_id_filter.removeprefix("gt.") if last_id_filter else None
    page = [row for row in rows if last_id is None or str(row["doc_id"]) > last_id]
    return httpx.Response(200, json=page[: int(params["limit"])])


def client_for(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_capture_uses_stable_keyset_pagination_and_checksums_each_collection(
    tmp_path: Path,
) -> None:
    rows = [api_document("a"), api_document("b"), api_document("c")]
    page_filters: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if (
            request.url.path == "/docs"
            and request.headers.get("Prefer") != "count=exact"
        ):
            page_filters.append(request.url.params.get("doc_id"))
        return response_for(request, rows=rows)

    manifest = capture_historylab_corpus(
        "https://api.example.test",
        tmp_path,
        client=client_for(handler),
        page_size=2,
        corpora=["cfpf"],
        captured_at="2026-11-12T12:00:00Z",
    )

    artifact = manifest.artifact_for("cfpf")
    output = tmp_path / artifact.path
    assert page_filters == [None, "gt.b"]
    assert [json.loads(line)["doc_id"] for line in output.read_text().splitlines()] == [
        "a",
        "b",
        "c",
    ]
    assert artifact.document_count == 3
    assert artifact.sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert manifest.counts_before == {"cfpf": 3}
    assert manifest.counts_after == {"cfpf": 3}
    assert manifest.complete is True
    assert (tmp_path / "capture-manifest.json").is_file()
    assert not (tmp_path / "capture.incomplete.json").exists()


def test_interrupted_capture_resumes_after_the_last_committed_id_without_duplicates(
    tmp_path: Path,
) -> None:
    rows = [api_document("a"), api_document("b"), api_document("c")]
    page_calls = [0]

    def interrupted(request: httpx.Request) -> httpx.Response:
        if (
            request.url.path == "/docs"
            and request.headers.get("Prefer") != "count=exact"
        ):
            page_calls[0] += 1
            if page_calls[0] == 2:
                raise httpx.ConnectError("temporary outage", request=request)
        return response_for(request, rows=rows)

    with pytest.raises(CorpusCaptureError, match="temporary outage"):
        capture_historylab_corpus(
            "https://api.example.test",
            tmp_path,
            client=client_for(interrupted),
            page_size=2,
            corpora=["cfpf"],
        )

    assert (tmp_path / "capture.incomplete.json").is_file()
    assert (tmp_path / "collections" / "cfpf.jsonl.part").is_file()
    resumed_filters: list[str | None] = []

    def resumed(request: httpx.Request) -> httpx.Response:
        if (
            request.url.path == "/docs"
            and request.headers.get("Prefer") != "count=exact"
        ):
            resumed_filters.append(request.url.params.get("doc_id"))
        return response_for(request, rows=rows)

    manifest = capture_historylab_corpus(
        "https://api.example.test",
        tmp_path,
        client=client_for(resumed),
        page_size=2,
        corpora=["cfpf"],
        resume=True,
    )

    assert resumed_filters == ["gt.b"]
    output = tmp_path / manifest.artifact_for("cfpf").path
    assert [json.loads(line)["doc_id"] for line in output.read_text().splitlines()] == [
        "a",
        "b",
        "c",
    ]


def test_capture_rejects_duplicate_or_non_increasing_document_ids(
    tmp_path: Path,
) -> None:
    rows = [api_document("a"), api_document("a")]

    with pytest.raises(CorpusCaptureError, match="strictly increasing"):
        capture_historylab_corpus(
            "https://api.example.test",
            tmp_path,
            client=client_for(lambda request: response_for(request, rows=rows)),
            page_size=10,
            corpora=["cfpf"],
        )


def test_capture_rejects_missing_rows_even_when_the_api_count_is_stable(
    tmp_path: Path,
) -> None:
    rows = [api_document("a"), api_document("b")]

    with pytest.raises(CorpusCaptureError, match="captured 2.*expected 3"):
        capture_historylab_corpus(
            "https://api.example.test",
            tmp_path,
            client=client_for(
                lambda request: response_for(request, rows=rows, count=3)
            ),
            page_size=10,
            corpora=["cfpf"],
        )


def test_capture_rejects_api_count_drift_and_keeps_partial_output_incomplete(
    tmp_path: Path,
) -> None:
    rows = [api_document("a"), api_document("b")]
    count_call = [0]

    with pytest.raises(CorpusCaptureError, match="changed during capture"):
        capture_historylab_corpus(
            "https://api.example.test",
            tmp_path,
            client=client_for(
                lambda request: response_for(
                    request,
                    rows=rows,
                    count=lambda index: 2 if index == 0 else 3,
                    count_call=count_call,
                )
            ),
            page_size=10,
            corpora=["cfpf"],
        )

    assert (tmp_path / "capture.incomplete.json").is_file()
    assert not (tmp_path / "capture-manifest.json").exists()


def test_capture_rejects_unsafe_collection_names_before_creating_collection_files(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/corpora"
        return httpx.Response(200, json=[{"corpus": "../escape", "doc_cnt": 1}])

    with pytest.raises(CorpusCaptureError, match="unsafe corpus identifier"):
        capture_historylab_corpus(
            "https://api.example.test",
            tmp_path,
            client=client_for(handler),
        )

    assert not (tmp_path.parent / "escape.jsonl").exists()


def test_capture_manifest_rejects_duplicate_collection_artifacts() -> None:
    artifact = ApiCaptureArtifact(
        corpus="cfpf",
        path="collections/cfpf.jsonl",
        document_count=1,
        size_bytes=1,
        sha256="a" * 64,
        first_doc_id="a",
        last_doc_id="a",
    )

    with pytest.raises(ValueError, match="unique"):
        ApiCaptureManifest(
            schema_version="historylab-postgrest-v1",
            base_url="https://api.example.test",
            captured_at="2026-11-12T12:00:00Z",
            select_fields=list(CAPTURE_SELECT_FIELDS),
            order="doc_id.asc",
            catalog_path="catalog.json",
            catalog_size_bytes=1,
            catalog_sha256="b" * 64,
            counts_before={"cfpf": 1},
            counts_after={"cfpf": 1},
            artifacts=[artifact, artifact],
            complete=True,
        )
