from __future__ import annotations

import asyncio
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

import redacted_backend.api.context as context_api
from redacted_backend.config import Settings
from redacted_backend.db.base import Base
from redacted_backend.db.models import VisibleExampleArtifact
from redacted_backend.db.repositories.documents import DocumentRepository
from redacted_backend.db.repositories.passages import PassageRepository
from redacted_backend.documents.preparation import PreparedVisibleContext
from redacted_backend.domain.documents import DateProvenance
from redacted_backend.domain.passages import PassageRect
from redacted_backend.domain.visible_examples import LengthEstimate, VisibleExample
from redacted_backend.main import create_app


async def request(app, method: str, path: str, **kwargs):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        return await client.request(method, path, **kwargs)


def create_test_app_and_passage(tmp_path: Path, *, with_date: bool = True):
    app = create_app(
        Settings(
            database_url=f"sqlite+pysqlite:///{tmp_path / 'context.sqlite3'}",
            document_storage_root=tmp_path / "stored-documents",
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
        if with_date:
            document = documents.set_visible_date(
                document.id,
                visible_date=date(1945, 5, 8),
                provenance=DateProvenance.MANUAL_VISIBLE_DOCUMENT,
            )
            assert document is not None
        passage = PassageRepository(session).create(
            document_id=document.id,
            label="Passage 1",
            page_number=1,
            rects=[PassageRect(x=0.3, y=0.2, width=0.2, height=0.04)],
        )
    return app, document, passage


def install_fake_preparation(monkeypatch, *, calls: list[int]) -> None:
    def fake_prepare(_pdf_path, document, _passages, target_id):
        calls.append(1)
        return PreparedVisibleContext(
            example=VisibleExample(
                passage_id=target_id,
                document_id=document.id,
                document_date=document.visible_date,
                before_target="visible before",
                after_target="visible after",
                context_window="visible before [[TARGET]] visible after",
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
        )

    monkeypatch.setattr(context_api, "prepare_visible_context", fake_prepare)


def test_prepare_requires_a_visible_document_date(tmp_path: Path, monkeypatch) -> None:
    app, _document, passage = create_test_app_and_passage(
        tmp_path,
        with_date=False,
    )
    calls: list[int] = []
    install_fake_preparation(monkeypatch, calls=calls)

    response = asyncio.run(
        request(app, "POST", f"/v1/passages/{passage.id}:prepare-context")
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "missing_visible_date"
    assert calls == []


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
    }
    assert document.visible_date == date(1945, 5, 8)
    assert calls == [1]
    with app.state.database.session_factory() as session:
        artifacts = session.scalars(select(VisibleExampleArtifact)).all()
        assert len(artifacts) == 1


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
        "prediction",
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
        assert artifacts[1].invalidated_at is None


def test_document_date_change_invalidates_prepared_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document, passage = create_test_app_and_passage(tmp_path)
    calls: list[int] = []
    install_fake_preparation(monkeypatch, calls=calls)
    path = f"/v1/passages/{passage.id}:prepare-context"
    assert asyncio.run(request(app, "POST", path)).status_code == 200

    metadata = asyncio.run(
        request(
            app,
            "PATCH",
            f"/v1/documents/{document.id}/metadata",
            json={
                "visible_date": "1945-05-09",
                "date_provenance": "manual_visible_document",
            },
        )
    )
    rebuilt = asyncio.run(request(app, "POST", path))

    assert metadata.status_code == 200
    assert rebuilt.status_code == 200
    assert rebuilt.json()["document_date"] == "1945-05-09"
    assert rebuilt.json()["passage_revision"] == 2
    assert calls == [1, 1]


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
