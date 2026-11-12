from __future__ import annotations

import asyncio
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from redacted_backend.config import Settings
from redacted_backend.db.base import Base
from redacted_backend.main import create_app
from tests.pdf_fixtures import make_pdf


def test_identical_uploads_share_bytes_but_keep_independent_records(
    tmp_path: Path,
) -> None:
    app = create_app(
        Settings(
            database_url=f"sqlite+pysqlite:///{tmp_path / 'documents.sqlite3'}",
            document_storage_root=tmp_path / "stored-documents",
        )
    )
    Base.metadata.create_all(app.state.database.engine)
    pdf_bytes = make_pdf()

    async def run_assertions() -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            first = await client.post(
                "/v1/documents",
                files={"file": ("first.pdf", pdf_bytes, "application/pdf")},
            )
            second = await client.post(
                "/v1/documents",
                files={"file": ("second.pdf", pdf_bytes, "application/pdf")},
            )
            assert first.status_code == 201
            assert second.status_code == 201
            first_body = first.json()
            second_body = second.json()
            assert first_body["id"] != second_body["id"]
            assert first_body["storage_key"] == second_body["storage_key"]

            stored_path = app.state.document_store.resolve(first_body["storage_key"])
            assert stored_path.read_bytes() == pdf_bytes
            assert len(list((tmp_path / "stored-documents").rglob("*.pdf"))) == 1

            assert (
                await client.delete(f"/v1/documents/{first_body['id']}")
            ).status_code == 204
            assert stored_path.is_file()

            assert (
                await client.delete(f"/v1/documents/{second_body['id']}")
            ).status_code == 204
            assert not stored_path.exists()

    asyncio.run(run_assertions())
