from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from redacted_backend.config import Settings
from redacted_backend.db.base import Base
from redacted_backend.db.models import Passage
from redacted_backend.db.repositories.documents import DocumentRepository
from redacted_backend.main import create_app


def create_test_app_and_document(tmp_path: Path):
    app = create_app(
        Settings(
            database_url=f"sqlite+pysqlite:///{tmp_path / 'passages.sqlite3'}",
            document_storage_root=tmp_path / "stored-documents",
        )
    )
    Base.metadata.create_all(app.state.database.engine)
    with app.state.database.session_factory() as session, session.begin():
        document = DocumentRepository(session).create(
            filename="visible-bulletin.pdf",
            sha256="a" * 64,
            media_type="application/pdf",
            page_count=3,
            storage_key=f"sha256/aa/aa/{'a' * 64}.pdf",
            expires_at=datetime.now(UTC) + timedelta(hours=24),
        )
    return app, str(document.id)


async def request(app, method: str, path: str, **kwargs):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        return await client.request(method, path, **kwargs)


def passage_body(*, y: float = 0.2, rects: list[dict] | None = None) -> dict:
    return {
        "label": "Passage 1",
        "page_number": 1,
        "rects": rects or [{"x": 0.1, "y": y, "width": 0.4, "height": 0.05}],
    }


def create_passage(app, document_id: str, *, y: float, key: str) -> dict:
    response = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/documents/{document_id}/passages",
            headers={"Idempotency-Key": key},
            json=passage_body(y=y),
        )
    )
    assert response.status_code == 201
    return response.json()


@pytest.mark.parametrize(
    "rect",
    [
        {"x": -0.01, "y": 0.1, "width": 0.2, "height": 0.1},
        {"x": 0.9, "y": 0.1, "width": 0.2, "height": 0.1},
        {"x": 0.1, "y": 0.95, "width": 0.2, "height": 0.1},
        {"x": 0.1, "y": 0.1, "width": 0.0, "height": 0.1},
    ],
)
def test_create_rejects_rectangles_outside_the_normalized_page(
    tmp_path: Path,
    rect: dict,
) -> None:
    app, document_id = create_test_app_and_document(tmp_path)

    response = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/documents/{document_id}/passages",
            json=passage_body(rects=[rect]),
        )
    )

    assert response.status_code == 422


def test_create_rejects_a_page_beyond_the_uploaded_document(tmp_path: Path) -> None:
    app, document_id = create_test_app_and_document(tmp_path)
    body = passage_body()
    body["page_number"] = 4

    response = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/documents/{document_id}/passages",
            json=body,
        )
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "passage_page_out_of_range"


def test_patch_keeps_the_passage_uuid_and_increments_revision(tmp_path: Path) -> None:
    app, document_id = create_test_app_and_document(tmp_path)
    original = create_passage(app, document_id, y=0.2, key="create-patch")

    response = asyncio.run(
        request(
            app,
            "PATCH",
            f"/v1/passages/{original['id']}",
            json={"rects": [{"x": 0.15, "y": 0.25, "width": 0.35, "height": 0.06}]},
        )
    )

    assert response.status_code == 200
    corrected = response.json()
    assert corrected["id"] == original["id"]
    assert corrected["revision"] == original["revision"] + 1
    assert corrected["status"] == "marked"


def test_merge_keeps_the_first_passage_in_document_order(tmp_path: Path) -> None:
    app, document_id = create_test_app_and_document(tmp_path)
    later = create_passage(app, document_id, y=0.7, key="create-later")
    earlier = create_passage(app, document_id, y=0.2, key="create-earlier")

    response = asyncio.run(
        request(
            app,
            "POST",
            "/v1/passages:merge",
            json={"passage_ids": [later["id"], earlier["id"]]},
        )
    )

    assert response.status_code == 200
    result = response.json()
    assert result["passage"]["id"] == earlier["id"]
    assert result["passage"]["revision"] == earlier["revision"] + 1
    assert result["merged_ids"] == [later["id"]]
    with app.state.database.session_factory() as session:
        merged = session.get(Passage, later["id"])
        assert merged is not None
        assert merged.merged_into == earlier["id"]
        assert merged.revision == later["revision"] + 1


def test_split_keeps_the_original_uuid_for_the_first_result(tmp_path: Path) -> None:
    app, document_id = create_test_app_and_document(tmp_path)
    original = create_passage(app, document_id, y=0.2, key="create-split")

    response = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/passages/{original['id']}:split",
            json={
                "parts": [
                    passage_body(y=0.2),
                    passage_body(y=0.35),
                ]
            },
        )
    )

    assert response.status_code == 200
    passages = response.json()["passages"]
    assert passages[0]["id"] == original["id"]
    assert passages[0]["revision"] == original["revision"] + 1
    assert passages[1]["id"] != original["id"]
    assert passages[1]["split_from"] == original["id"]
    assert passages[1]["revision"] == 1


def test_delete_tombstones_the_passage_and_increments_revision(tmp_path: Path) -> None:
    app, document_id = create_test_app_and_document(tmp_path)
    original = create_passage(app, document_id, y=0.2, key="create-delete")

    response = asyncio.run(request(app, "DELETE", f"/v1/passages/{original['id']}"))

    assert response.status_code == 200
    deleted = response.json()
    assert deleted["id"] == original["id"]
    assert deleted["status"] == "deleted"
    assert deleted["revision"] == original["revision"] + 1


def test_deleting_a_document_tombstones_its_live_passages(tmp_path: Path) -> None:
    app, document_id = create_test_app_and_document(tmp_path)
    original = create_passage(app, document_id, y=0.2, key="create-document-delete")

    response = asyncio.run(request(app, "DELETE", f"/v1/documents/{document_id}"))

    assert response.status_code == 204
    with app.state.database.session_factory() as session:
        persisted = session.get(Passage, original["id"])
        assert persisted is not None
        assert persisted.status == "deleted"
        assert persisted.revision == original["revision"] + 1


def test_reusing_an_idempotency_key_returns_one_passage_record(tmp_path: Path) -> None:
    app, document_id = create_test_app_and_document(tmp_path)
    headers = {"Idempotency-Key": "same-create-click"}

    first = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/documents/{document_id}/passages",
            headers=headers,
            json=passage_body(y=0.2),
        )
    )
    second = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/documents/{document_id}/passages",
            headers=headers,
            json=passage_body(y=0.2),
        )
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["id"] == second.json()["id"]
    with app.state.database.session_factory() as session:
        count = session.scalar(select(func.count()).select_from(Passage))
        assert count == 1


def test_reusing_an_idempotency_key_with_different_geometry_conflicts(
    tmp_path: Path,
) -> None:
    app, document_id = create_test_app_and_document(tmp_path)
    headers = {"Idempotency-Key": "conflicting-create"}
    first = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/documents/{document_id}/passages",
            headers=headers,
            json=passage_body(y=0.2),
        )
    )

    second = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/documents/{document_id}/passages",
            headers=headers,
            json=passage_body(y=0.4),
        )
    )

    assert first.status_code == 201
    assert second.status_code == 409
