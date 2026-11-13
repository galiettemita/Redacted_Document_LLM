from __future__ import annotations

from pathlib import Path

from redacted_backend.db.repositories.analysis import AnalysisJobRepository
from redacted_backend.db.repositories.passages import PassageRepository
from redacted_backend.domain.passages import PassageRect
from tests.api.test_context import create_test_app_and_passage


def test_passage_correction_cancels_active_job_and_orders_a_terminal_event(
    tmp_path: Path,
) -> None:
    app, _document, passage = create_test_app_and_passage(tmp_path)
    with app.state.database.session_factory() as session, session.begin():
        jobs = AnalysisJobRepository(session)
        job = jobs.schedule(
            passage_id=passage.id,
            document_id=passage.document_id,
            passage_revision=passage.revision,
            idempotency_key="queued-job",
            blockers=[],
        )
        PassageRepository(session).patch(
            passage.id,
            label=None,
            page_number=1,
            rects=[PassageRect(x=0.1, y=0.1, width=0.2, height=0.05)],
        )

    with app.state.database.session_factory() as session:
        jobs = AnalysisJobRepository(session)
        cancelled = jobs.get(job.id)
        events = jobs.events_after(passage.document_id, 0)

    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert cancelled.error_code == "passage_revision_changed"
    assert [event.sequence for event in events] == [1, 2]
    assert events[-1].payload["status"] == "cancelled"
