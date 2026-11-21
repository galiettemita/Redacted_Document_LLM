from __future__ import annotations

import json
from pathlib import Path

from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from alembic import command


def test_upgrade_backfills_known_legacy_prediction_basis_without_losing_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "prediction-basis.db"
    database_url = f"sqlite+pysqlite:///{database_path}"
    monkeypatch.setenv("REDACTED_DATABASE_URL", database_url)
    config = Config(str(Path(__file__).parents[2] / "alembic.ini"))
    command.upgrade(config, "0009_storage_cleanup_state")

    engine = create_engine(database_url)
    document_id = "00000000-0000-0000-0000-000000000001"
    passage_id = "00000000-0000-0000-0000-000000000002"
    job_id = "00000000-0000-0000-0000-000000000003"
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO documents "
                "(id, filename, sha256, media_type, page_count, status, "
                "storage_key, expires_at, mask_set_revision) VALUES "
                "(:id, 'legacy.pdf', :sha, 'application/pdf', 1, 'ready', "
                ":storage_key, '2026-11-21 12:00:00', 0)"
            ),
            {
                "id": document_id,
                "sha": "a" * 64,
                "storage_key": "sha256/aa/aa/legacy.pdf",
            },
        )
        connection.execute(
            text(
                "INSERT INTO passages "
                "(id, document_id, label, page_number, rects, source, status, "
                "revision) VALUES (:id, :document_id, 'Passage 1', 1, :rects, "
                "'manual', 'complete', 1)"
            ),
            {
                "id": passage_id,
                "document_id": document_id,
                "rects": '[{"x":0.1,"y":0.1,"width":0.2,"height":0.1}]',
            },
        )
        connection.execute(
            text(
                "INSERT INTO analysis_jobs "
                "(id, passage_id, document_id, passage_revision, "
                "idempotency_key, request_hash, status, stage, blockers, "
                "prediction_payload) VALUES "
                "(:id, :passage_id, :document_id, 1, 'legacy-key', :hash, "
                "'complete', 'complete', '[]', :prediction)"
            ),
            {
                "id": job_id,
                "passage_id": passage_id,
                "document_id": document_id,
                "hash": "b" * 64,
                "prediction": json.dumps(
                    {
                        "prediction": "Legacy exact-paper prediction",
                        "retrieval_trace_sha256": "c" * 64,
                    }
                ),
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
                "job_id": job_id,
                "document_id": document_id,
                "passage_id": passage_id,
                "payload": json.dumps(
                    {"job_id": job_id, "prediction": "Legacy event"}
                ),
            },
        )

    command.upgrade(config, "head")

    assert "prediction_basis" in {
        column["name"] for column in inspect(engine).get_columns("analysis_jobs")
    }
    with engine.connect() as connection:
        job = connection.execute(
            text(
                "SELECT prediction_basis, prediction_payload "
                "FROM analysis_jobs WHERE id = :id"
            ),
            {"id": job_id},
        ).one()
        event_payload = connection.scalar(
            text("SELECT payload FROM analysis_events WHERE job_id = :job_id"),
            {"job_id": job_id},
        )
    assert job.prediction_basis == "paper_exact_v1"
    assert _json(job.prediction_payload)["prediction_basis"] == "paper_exact_v1"
    assert _json(event_payload)["prediction_basis"] == "paper_exact_v1"

    command.downgrade(config, "0009_storage_cleanup_state")

    assert "prediction_basis" not in {
        column["name"] for column in inspect(engine).get_columns("analysis_jobs")
    }
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM analysis_jobs")) == 1
        assert connection.scalar(text("SELECT count(*) FROM analysis_events")) == 1


def _json(value: object) -> dict[str, object]:
    if isinstance(value, str):
        return json.loads(value)
    assert isinstance(value, dict)
    return value
