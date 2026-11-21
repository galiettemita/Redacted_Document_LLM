from pathlib import Path

from redacted_backend.db.repositories.analysis import AnalysisJobRepository
from redacted_backend.domain.analysis import PredictionBasis
from redacted_backend.jobs.generation_store import AnalysisGenerationStore
from redacted_backend.models.openai_gateway import StoredGeneration
from tests.api.test_context import create_test_app_and_passage


def test_generation_store_survives_a_new_worker_process_boundary(
    tmp_path: Path,
) -> None:
    app, document, passage = create_test_app_and_passage(tmp_path)
    with app.state.database.session_factory() as session, session.begin():
        job = AnalysisJobRepository(session).schedule(
            passage_id=passage.id,
            document_id=document.id,
            passage_revision=passage.revision,
            idempotency_key="durable-generation",
            blockers=[],
            prediction_basis=PredictionBasis.PAPER_EXACT_V1,
        )
    value = StoredGeneration(
        kind="query",
        request_sha256="a" * 64,
        payload={"request_sha256": "a" * 64, "query": "visible terms"},
    )

    store = AnalysisGenerationStore(app.state.database)
    assert store.claim(
        f"query:{passage.id}",
        f"{job.id}:query",
        "a" * 64,
        "query",
    )
    claimed = AnalysisGenerationStore(app.state.database).get(
        f"query:{passage.id}",
        f"{job.id}:query",
    )
    assert claimed is not None
    assert claimed.state == "claimed"
    assert not AnalysisGenerationStore(app.state.database).claim(
        f"query:{passage.id}",
        f"{job.id}:query",
        "a" * 64,
        "query",
    )

    store.save(
        f"query:{passage.id}",
        f"{job.id}:query",
        value,
    )

    restored = AnalysisGenerationStore(app.state.database).get(
        f"query:{passage.id}",
        f"{job.id}:query",
    )
    assert restored == value
