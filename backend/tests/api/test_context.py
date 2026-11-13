from __future__ import annotations

import asyncio
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

import redacted_backend.api.context as context_api
from redacted_backend.config import Settings
from redacted_backend.db.base import Base
from redacted_backend.db.models import Document, VisibleExampleArtifact
from redacted_backend.db.repositories.documents import DocumentRepository
from redacted_backend.db.repositories.passages import PassageRepository
from redacted_backend.documents.preparation import PreparedVisibleContext
from redacted_backend.domain.documents import DateProvenance, VisibleDateEvidence
from redacted_backend.domain.passages import PassageRect
from redacted_backend.domain.visible_examples import (
    LengthEstimate,
    PageWord,
    VisibleExample,
)
from redacted_backend.main import create_app

CONTEXT_QUERY = "visible before visible after"
CONTEXT_QUERY_SHA256 = (
    "4e64699ee5403e5b9c960206e944cdfde3be184499c2552b508afdaf089746fb"
)


async def request(app, method: str, path: str, **kwargs):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        return await client.request(method, path, **kwargs)


def create_test_app_and_passage(tmp_path: Path):
    app = create_app(
        Settings(
            database_url=f"sqlite+pysqlite:///{tmp_path / 'context.sqlite3'}",
            document_storage_root=tmp_path / "stored-documents",
            opensearch_url="http://127.0.0.1:1",
        )
    )
    Base.metadata.create_all(app.state.database.engine)
    with app.state.database.session_factory() as session, session.begin():
        documents = DocumentRepository(session)
        document = documents.create(
            filename="visible-bulletin.pdf",
            sha256="a" * 64,
            media_type="application/pdf",
            page_count=1,
            storage_key=f"sha256/aa/aa/{'a' * 64}.pdf",
            expires_at=datetime.now(UTC) + timedelta(hours=24),
        )
        passage = PassageRepository(session).create(
            document_id=document.id,
            label="Passage 1",
            page_number=1,
            rects=[PassageRect(x=0.3, y=0.2, width=0.2, height=0.04)],
        )
    return app, document, passage


def install_fake_preparation(
    monkeypatch,
    *,
    calls: list[int],
    context_window: str = "visible before [[TARGET]] visible after",
) -> None:
    def fake_prepare(_pdf_path, document, _passages, target_id):
        calls.append(1)
        return PreparedVisibleContext(
            example=VisibleExample(
                passage_id=target_id,
                document_id=document.id,
                document_date=date(1945, 5, 8),
                before_target="visible before",
                after_target="visible after",
                context_window=context_window,
                length_cue=LengthEstimate(
                    estimated_words=8,
                    bucket="part of a sentence",
                    median_character_width=0.01,
                    median_characters_per_word=5,
                    line_capacities=[48],
                ),
                visible_words=[],
                serialization_version="visible-transcript-v1",
            ),
            confirmed_mask_count=2,
            visible_word_count=17,
            date_evidence=_date_evidence(date(1945, 5, 8)),
        )

    monkeypatch.setattr(context_api, "prepare_visible_context", fake_prepare)


def _date_evidence(value: date) -> VisibleDateEvidence:
    return VisibleDateEvidence(
        value=value,
        page_number=1,
        line_text=f"DATE: {value.isoformat()}",
        words=[
            PageWord(
                text=value.isoformat(),
                page_number=1,
                x=0.1,
                y=0.08,
                width=0.15,
                height=0.02,
                block_id=0,
                line_id=1,
                source="embedded",
                confidence=1.0,
            )
        ],
    )


def test_prepare_detects_and_persists_the_visible_document_date(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document, passage = create_test_app_and_passage(tmp_path)
    calls: list[int] = []
    install_fake_preparation(monkeypatch, calls=calls)

    response = asyncio.run(
        request(app, "POST", f"/v1/passages/{passage.id}:prepare-context")
    )

    assert response.status_code == 200
    assert response.json()["document_date"] == "1945-05-08"
    assert calls == [1]
    with app.state.database.session_factory() as session:
        persisted = DocumentRepository(session).get(document.id)
        assert persisted is not None
        assert persisted.visible_date == date(1945, 5, 8)
        assert persisted.date_provenance == DateProvenance.VISIBLE_TEXT
        assert persisted.visible_date_evidence == _date_evidence(date(1945, 5, 8))


def test_prepare_persists_one_revision_and_reuses_it(
    tmp_path: Path, monkeypatch
) -> None:
    app, document, passage = create_test_app_and_passage(tmp_path)
    calls: list[int] = []
    install_fake_preparation(monkeypatch, calls=calls)

    first = asyncio.run(
        request(app, "POST", f"/v1/passages/{passage.id}:prepare-context")
    )
    second = asyncio.run(
        request(app, "POST", f"/v1/passages/{passage.id}:prepare-context")
    )

    assert first.status_code == 200
    assert second.status_code == 200
    summary = first.json()
    assert summary == second.json()
    assert summary == {
        "passage_id": str(passage.id),
        "passage_revision": 1,
        "status": "context_ready",
        "document_date": "1945-05-08",
        "length_cue": "part of a sentence",
        "estimated_words": 8,
        "visible_word_count": 17,
        "confirmed_mask_count": 2,
        "serialization_version": "visible-transcript-v1",
        "context_query_status": "ready",
        "context_query_term_count": 4,
        "context_query_sha256": CONTEXT_QUERY_SHA256,
        "context_query_version": "context-query-v1",
        "context_stopwords_version": "visible-english-v1",
    }
    assert document.visible_date is None
    assert calls == [1]
    with app.state.database.session_factory() as session:
        artifacts = session.scalars(select(VisibleExampleArtifact)).all()
        assert len(artifacts) == 1
        assert artifacts[0].context_query == CONTEXT_QUERY
        assert artifacts[0].context_query_sha256 == CONTEXT_QUERY_SHA256
        assert artifacts[0].context_query_term_count == 4
        assert artifacts[0].context_query_version == "context-query-v1"
        assert artifacts[0].context_stopwords_version == "visible-english-v1"


def test_prepare_backfills_query_metadata_on_a_same_revision_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, _document, passage = create_test_app_and_passage(tmp_path)
    calls: list[int] = []
    install_fake_preparation(monkeypatch, calls=calls)
    path = f"/v1/passages/{passage.id}:prepare-context"
    assert asyncio.run(request(app, "POST", path)).status_code == 200

    with app.state.database.session_factory() as session, session.begin():
        artifact = session.scalar(select(VisibleExampleArtifact))
        assert artifact is not None
        artifact.context_query = None
        artifact.context_query_sha256 = None
        artifact.context_query_term_count = None
        artifact.context_query_version = None
        artifact.context_stopwords_version = None

    response = asyncio.run(request(app, "POST", path))

    assert response.status_code == 200
    assert response.json()["context_query_sha256"] == CONTEXT_QUERY_SHA256
    assert calls == [1]
    with app.state.database.session_factory() as session:
        artifact = session.scalar(select(VisibleExampleArtifact))
        assert artifact is not None
        assert artifact.context_query == CONTEXT_QUERY
        assert artifact.context_query_term_count == 4


def test_prepare_rejects_context_without_visible_search_terms_before_persisting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document, passage = create_test_app_and_passage(tmp_path)
    calls: list[int] = []
    install_fake_preparation(
        monkeypatch,
        calls=calls,
        context_window="The [[TARGET]] and [REDACTED]",
    )

    response = asyncio.run(
        request(app, "POST", f"/v1/passages/{passage.id}:prepare-context")
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "context_query_empty"
    assert calls == [1]
    with app.state.database.session_factory() as session:
        assert session.scalar(select(VisibleExampleArtifact)) is None
        persisted = DocumentRepository(session).get(document.id)
        assert persisted is not None
        assert persisted.visible_date is None


def test_prepare_refuses_incomplete_query_metadata_instead_of_overwriting_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, _document, passage = create_test_app_and_passage(tmp_path)
    calls: list[int] = []
    install_fake_preparation(monkeypatch, calls=calls)
    path = f"/v1/passages/{passage.id}:prepare-context"
    assert asyncio.run(request(app, "POST", path)).status_code == 200

    with app.state.database.session_factory() as session, session.begin():
        artifact = session.scalar(select(VisibleExampleArtifact))
        assert artifact is not None
        artifact.context_query_sha256 = None

    response = asyncio.run(request(app, "POST", path))

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "context_query_conflict"
    assert calls == [1]
    with app.state.database.session_factory() as session:
        artifact = session.scalar(select(VisibleExampleArtifact))
        assert artifact is not None
        assert artifact.context_query == CONTEXT_QUERY
        assert artifact.context_query_sha256 is None


def test_prepare_refuses_fully_populated_query_metadata_that_does_not_match(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, _document, passage = create_test_app_and_passage(tmp_path)
    calls: list[int] = []
    install_fake_preparation(monkeypatch, calls=calls)
    path = f"/v1/passages/{passage.id}:prepare-context"
    assert asyncio.run(request(app, "POST", path)).status_code == 200

    with app.state.database.session_factory() as session, session.begin():
        artifact = session.scalar(select(VisibleExampleArtifact))
        assert artifact is not None
        artifact.context_query_sha256 = "0" * 64

    response = asyncio.run(request(app, "POST", path))

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "context_query_conflict"
    assert calls == [1]
    with app.state.database.session_factory() as session:
        artifact = session.scalar(select(VisibleExampleArtifact))
        assert artifact is not None
        assert artifact.context_query_sha256 == "0" * 64


@pytest.mark.parametrize("identity_field", ("passage_id", "document_id"))
def test_prepare_refuses_legacy_payload_bound_to_a_different_identity(
    tmp_path: Path,
    monkeypatch,
    identity_field: str,
) -> None:
    app, _document, passage = create_test_app_and_passage(tmp_path)
    calls: list[int] = []
    install_fake_preparation(monkeypatch, calls=calls)
    path = f"/v1/passages/{passage.id}:prepare-context"
    assert asyncio.run(request(app, "POST", path)).status_code == 200

    with app.state.database.session_factory() as session, session.begin():
        artifact = session.scalar(select(VisibleExampleArtifact))
        assert artifact is not None
        payload = dict(artifact.payload)
        payload[identity_field] = str(uuid4())
        artifact.payload = payload
        _clear_query_metadata(artifact)

    response = asyncio.run(request(app, "POST", path))

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "context_query_conflict"
    assert calls == [1]
    with app.state.database.session_factory() as session:
        artifact = session.scalar(select(VisibleExampleArtifact))
        assert artifact is not None
        assert artifact.context_query is None


@pytest.mark.parametrize(
    "context_window",
    (
        "visible before visible after",
        "visible [[TARGET]] middle [[TARGET]] after",
    ),
)
def test_prepare_refuses_legacy_payload_without_exactly_one_target_marker(
    tmp_path: Path,
    monkeypatch,
    context_window: str,
) -> None:
    app, _document, passage = create_test_app_and_passage(tmp_path)
    calls: list[int] = []
    install_fake_preparation(monkeypatch, calls=calls)
    path = f"/v1/passages/{passage.id}:prepare-context"
    assert asyncio.run(request(app, "POST", path)).status_code == 200

    with app.state.database.session_factory() as session, session.begin():
        artifact = session.scalar(select(VisibleExampleArtifact))
        assert artifact is not None
        payload = dict(artifact.payload)
        payload["context_window"] = context_window
        artifact.payload = payload
        _clear_query_metadata(artifact)

    response = asyncio.run(request(app, "POST", path))

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "context_query_conflict"
    assert calls == [1]
    with app.state.database.session_factory() as session:
        artifact = session.scalar(select(VisibleExampleArtifact))
        assert artifact is not None
        assert artifact.context_query is None


def test_browser_summary_omits_transcript_prediction_and_hidden_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, _document, passage = create_test_app_and_passage(tmp_path)
    calls: list[int] = []
    install_fake_preparation(monkeypatch, calls=calls)

    response = asyncio.run(
        request(app, "POST", f"/v1/passages/{passage.id}:prepare-context")
    )

    response_text = json.dumps(response.json()).lower()
    for forbidden in (
        "before_target",
        "after_target",
        "context_window",
        CONTEXT_QUERY,
        "prediction",
        "evidence",
        "search_results",
        "answer",
        "ground_truth",
    ):
        assert forbidden not in response_text

    with app.state.database.session_factory() as session:
        artifact = session.scalar(select(VisibleExampleArtifact))
        assert artifact is not None
        payload = json.dumps(artifact.payload).lower()
        assert "visible before" in payload
        for forbidden in ("prediction", "answer", "ground_truth"):
            assert forbidden not in payload


def test_passage_correction_invalidates_context_and_builds_new_revision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, _document, passage = create_test_app_and_passage(tmp_path)
    calls: list[int] = []
    install_fake_preparation(monkeypatch, calls=calls)
    path = f"/v1/passages/{passage.id}:prepare-context"
    assert asyncio.run(request(app, "POST", path)).status_code == 200

    correction = asyncio.run(
        request(
            app,
            "PATCH",
            f"/v1/passages/{passage.id}",
            json={"rects": [{"x": 0.32, "y": 0.22, "width": 0.18, "height": 0.04}]},
        )
    )
    rebuilt = asyncio.run(request(app, "POST", path))

    assert correction.status_code == 200
    assert rebuilt.status_code == 200
    assert rebuilt.json()["passage_revision"] == 2
    assert calls == [1, 1]
    with app.state.database.session_factory() as session:
        artifacts = session.scalars(
            select(VisibleExampleArtifact).order_by(
                VisibleExampleArtifact.passage_revision
            )
        ).all()
        assert [artifact.passage_revision for artifact in artifacts] == [1, 2]
        assert artifacts[0].invalidated_at is not None
        assert artifacts[0].context_query_sha256 == CONTEXT_QUERY_SHA256
        assert artifacts[1].invalidated_at is None
        assert artifacts[1].context_query_sha256 == CONTEXT_QUERY_SHA256


def test_conflicting_automatic_date_is_refused_without_overwriting_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document, passage = create_test_app_and_passage(tmp_path)
    with app.state.database.session_factory() as session, session.begin():
        stored = DocumentRepository(session).set_detected_visible_date(
            document.id,
            _date_evidence(date(1945, 5, 9)),
        )
        assert stored is not None

    calls: list[int] = []
    install_fake_preparation(monkeypatch, calls=calls)
    response = asyncio.run(
        request(app, "POST", f"/v1/passages/{passage.id}:prepare-context")
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "visible_date_conflict"
    assert calls == [1]
    with app.state.database.session_factory() as session:
        persisted = DocumentRepository(session).get(document.id)
        assert persisted is not None
        assert persisted.visible_date == date(1945, 5, 9)
        assert persisted.visible_date_evidence == _date_evidence(date(1945, 5, 9))


def test_prepare_backfills_missing_evidence_for_a_matching_automatic_date(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document, passage = create_test_app_and_passage(tmp_path)
    with app.state.database.session_factory() as session, session.begin():
        stored = session.get(Document, str(document.id))
        assert stored is not None
        stored.visible_date = date(1945, 5, 8)
        stored.date_provenance = DateProvenance.VISIBLE_TEXT.value
        stored.visible_date_evidence = None

    calls: list[int] = []
    install_fake_preparation(monkeypatch, calls=calls)
    response = asyncio.run(
        request(app, "POST", f"/v1/passages/{passage.id}:prepare-context")
    )

    assert response.status_code == 200
    with app.state.database.session_factory() as session:
        persisted = DocumentRepository(session).get(document.id)
        assert persisted is not None
        assert persisted.visible_date_evidence == _date_evidence(date(1945, 5, 8))


def test_deleted_passage_cannot_prepare_context(tmp_path: Path, monkeypatch) -> None:
    app, _document, passage = create_test_app_and_passage(tmp_path)
    calls: list[int] = []
    install_fake_preparation(monkeypatch, calls=calls)

    deleted = asyncio.run(request(app, "DELETE", f"/v1/passages/{passage.id}"))
    response = asyncio.run(
        request(app, "POST", f"/v1/passages/{passage.id}:prepare-context")
    )

    assert deleted.status_code == 200
    assert response.status_code == 404
    assert calls == []


def _clear_query_metadata(artifact: VisibleExampleArtifact) -> None:
    artifact.context_query = None
    artifact.context_query_sha256 = None
    artifact.context_query_term_count = None
    artifact.context_query_version = None
    artifact.context_stopwords_version = None
