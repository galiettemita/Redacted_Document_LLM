from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from sqlalchemy import create_engine, text

from alembic import command


def test_upgrade_scrubs_sensitive_artifacts_from_existing_tombstones(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "retention-upgrade.db"
    database_url = f"sqlite+pysqlite:///{database_path}"
    monkeypatch.setenv("REDACTED_DATABASE_URL", database_url)
    config = Config(str(Path(__file__).parents[2] / "alembic.ini"))
    command.upgrade(config, "0008_document_retention")

    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO documents "
                "(id, filename, sha256, media_type, page_count, status, "
                "storage_key, visible_date, date_provenance, "
                "visible_date_evidence, expires_at, deleted_at, "
                "deletion_reason, mask_set_revision) VALUES "
                "(:id, 'legacy.pdf', :sha, 'application/pdf', 1, 'deleted', "
                ":storage_key, '1945-05-08', 'visible_text', :evidence, "
                "'2026-11-14 12:00:00', '2026-11-14 12:00:01', NULL, 0)"
            ),
            {
                "id": "00000000-0000-0000-0000-000000000001",
                "sha": "a" * 64,
                "storage_key": "sha256/aa/aa/legacy.pdf",
                "evidence": '{"line_text":"PRIVATE DATE LINE"}',
            },
        )
        connection.execute(
            text(
                "INSERT INTO passages "
                "(id, document_id, label, page_number, rects, source, status, "
                "revision) VALUES (:id, :document_id, 'Passage 1', 1, :rects, "
                "'manual', 'deleted', 1)"
            ),
            {
                "id": "00000000-0000-0000-0000-000000000002",
                "document_id": "00000000-0000-0000-0000-000000000001",
                "rects": '[{"x":0.1,"y":0.1,"width":0.2,"height":0.1}]',
            },
        )
        connection.execute(
            text(
                "INSERT INTO visible_example_artifacts "
                "(id, passage_id, document_id, passage_revision, payload, "
                "visible_word_count, confirmed_mask_count) VALUES "
                "(:id, :passage_id, :document_id, 1, :payload, 1, 1)"
            ),
            {
                "id": "00000000-0000-0000-0000-000000000003",
                "passage_id": "00000000-0000-0000-0000-000000000002",
                "document_id": "00000000-0000-0000-0000-000000000001",
                "payload": '{"context_window":"PRIVATE VISIBLE TEXT"}',
            },
        )
        connection.execute(
            text(
                "INSERT INTO analysis_jobs "
                "(id, passage_id, document_id, passage_revision, "
                "idempotency_key, request_hash, status, stage, blockers, "
                "query_payload, retrieval_trace, prediction_payload) VALUES "
                "(:id, :passage_id, :document_id, 1, 'legacy-key', :hash, "
                "'complete', 'complete', '[]', :query, :trace, :prediction)"
            ),
            {
                "id": "00000000-0000-0000-0000-000000000004",
                "passage_id": "00000000-0000-0000-0000-000000000002",
                "document_id": "00000000-0000-0000-0000-000000000001",
                "hash": "b" * 64,
                "query": '{"query":"PRIVATE QUERY"}',
                "trace": '{"evidence":"PRIVATE RETRIEVAL"}',
                "prediction": '{"prediction":"PRIVATE PREDICTION"}',
            },
        )
        connection.execute(
            text(
                "INSERT INTO analysis_events "
                "(job_id, document_id, passage_id, sequence, event_type, payload) "
                "VALUES (:job_id, :document_id, :passage_id, 1, "
                "'passage.prediction', :payload)"
            ),
            {
                "job_id": "00000000-0000-0000-0000-000000000004",
                "document_id": "00000000-0000-0000-0000-000000000001",
                "passage_id": "00000000-0000-0000-0000-000000000002",
                "payload": '{"prediction":"PRIVATE EVENT"}',
            },
        )

    command.upgrade(config, "head")

    with engine.connect() as connection:
        document = connection.execute(
            text(
                "SELECT visible_date, date_provenance, visible_date_evidence, "
                "deletion_reason, storage_deleted_at FROM documents"
            )
        ).one()
        assert document.visible_date is None
        assert document.date_provenance is None
        assert document.visible_date_evidence is None
        assert document.deletion_reason == "legacy_deleted"
        assert document.storage_deleted_at is None
        assert connection.scalar(text("SELECT count(*) FROM analysis_events")) == 0
        assert connection.scalar(text("SELECT count(*) FROM analysis_jobs")) == 0
        assert (
            connection.scalar(text("SELECT count(*) FROM visible_example_artifacts"))
            == 0
        )
