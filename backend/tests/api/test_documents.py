from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from uuid import UUID

from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from redacted_backend.config import Settings
from redacted_backend.db.base import Base
from redacted_backend.db.models import Document
from redacted_backend.main import create_app
from tests.pdf_fixtures import make_pdf


def create_test_app(tmp_path: Path):
    app = create_app(
        Settings(
            database_url=f"sqlite+pysqlite:///{tmp_path / 'documents.sqlite3'}",
            document_storage_root=tmp_path / "stored-documents",
        )
    )
    Base.metadata.create_all(app.state.database.engine)
    return app


async def request(app, method: str, path: str, **kwargs):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        return await client.request(method, path, **kwargs)


def test_upload_sanitizes_the_name_and_persists_only_a_sha256_storage_key(
    tmp_path: Path,
) -> None:
    app = create_test_app(tmp_path)
    pdf_bytes = make_pdf()
    expected_hash = hashlib.sha256(pdf_bytes).hexdigest()

    response = asyncio.run(
        request(
            app,
            "POST",
            "/v1/documents",
            files={
                "file": (
                    "../../Visible Bulletin.pdf",
                    pdf_bytes,
                    "application/pdf",
                )
            },
        )
    )

    assert response.status_code == 201
    body = response.json()
    UUID(body["id"])
    assert body["filename"] == "Visible Bulletin.pdf"
    assert body["sha256"] == expected_hash
    assert body["media_type"] == "application/pdf"
    assert body["page_count"] == 1
    assert body["status"] == "uploaded"
    assert body["storage_key"] == (
        f"sha256/{expected_hash[:2]}/{expected_hash[2:4]}/{expected_hash}.pdf"
    )
    assert "storage_path" not in body
    assert body["visible_date"] is None
    assert body["date_provenance"] is None


def test_upload_rejects_non_pdf_content_without_retaining_it(tmp_path: Path) -> None:
    app = create_test_app(tmp_path)

    response = asyncio.run(
        request(
            app,
            "POST",
            "/v1/documents",
            files={"file": ("fake.pdf", b"not a PDF", "application/pdf")},
        )
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_signature"
    assert list((tmp_path / "stored-documents").rglob("*.pdf")) == []


def test_browser_cannot_supply_document_date_metadata(
    tmp_path: Path,
) -> None:
    app = create_test_app(tmp_path)
    upload = asyncio.run(
        request(
            app,
            "POST",
            "/v1/documents",
            files={
                "file": ("bulletin.pdf", make_pdf(), "application/pdf"),
            },
        )
    )
    document_id = upload.json()["id"]

    response = asyncio.run(
        request(
            app,
            "PATCH",
            f"/v1/documents/{document_id}/metadata",
            json={
                "visible_date": "1945-05-08",
                "date_provenance": "manual_visible_document",
            },
        )
    )

    assert response.status_code == 404


def test_public_api_schema_has_no_manual_date_provenance(tmp_path: Path) -> None:
    schema = json.dumps(create_test_app(tmp_path).openapi())

    assert "manual_visible_document" not in schema


def test_delete_tombstones_the_record_and_removes_unreferenced_bytes(
    tmp_path: Path,
) -> None:
    app = create_test_app(tmp_path)
    upload = asyncio.run(
        request(
            app,
            "POST",
            "/v1/documents",
            files={"file": ("bulletin.pdf", make_pdf(), "application/pdf")},
        )
    )
    body = upload.json()
    stored_path = app.state.document_store.resolve(body["storage_key"])
    assert stored_path.is_file()

    response = asyncio.run(request(app, "DELETE", f"/v1/documents/{body['id']}"))

    assert response.status_code == 204
    assert not stored_path.exists()
    with app.state.database.session_factory() as session:
        record = session.scalar(select(Document).where(Document.id == body["id"]))
        assert record is not None
        assert record.status == "deleted"
