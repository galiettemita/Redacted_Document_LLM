from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

import redacted_backend.api.passages as passages_api
from redacted_backend.config import Settings
from redacted_backend.db.base import Base
from redacted_backend.db.models import Passage
from redacted_backend.documents.detection import DetectedPassage, DetectionResult
from redacted_backend.domain.passages import PassageRect, PassageSource, PassageStatus
from redacted_backend.main import create_app


async def request(app, method: str, path: str, **kwargs):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        return await client.request(method, path, **kwargs)


def test_detect_endpoint_persists_all_passages_and_reuses_stable_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document_id = _uploaded_document(tmp_path)
    monkeypatch.setattr(passages_api, "detect_passages", lambda _path: _result())

    first = asyncio.run(
        request(app, "POST", f"/v1/documents/{document_id}/passages:detect")
    )
    second = asyncio.run(
        request(app, "POST", f"/v1/documents/{document_id}/passages:detect")
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    body = first.json()
    assert body["document_id"] == document_id
    assert body["status"] == "detection_ready"
    assert body["pages_scanned"] == 2
    assert body["coverage_confident"] is True
    assert [item["label"] for item in body["passages"]] == [
        "Passage 1",
        "Passage 2",
    ]
    assert [item["source"] for item in body["passages"]] == [
        "detected",
        "detected",
    ]
    with app.state.database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(Passage)) == 2


def test_concurrent_sqlite_detection_requests_serialize_and_return_one_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document_id = _uploaded_document(tmp_path)
    state_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def slow_detection(_path: Path) -> DetectionResult:
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.1)
        with state_lock:
            active -= 1
        return _result(include_second=False)

    monkeypatch.setattr(passages_api, "detect_passages", slow_detection)

    async def run_both():
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            return await asyncio.gather(
                client.post(f"/v1/documents/{document_id}/passages:detect"),
                client.post(f"/v1/documents/{document_id}/passages:detect"),
            )

    first, second = asyncio.run(run_both())

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert maximum_active == 1
    with app.state.database.session_factory() as session:
        live_count = session.scalar(
            select(func.count())
            .select_from(Passage)
            .where(Passage.status != PassageStatus.DELETED.value)
        )
        assert live_count == 1


def test_redetection_tombstones_removed_geometry_without_changing_retained_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document_id = _uploaded_document(tmp_path)
    results = [_result(), _result(include_second=False)]
    monkeypatch.setattr(passages_api, "detect_passages", lambda _path: results.pop(0))

    first = asyncio.run(
        request(app, "POST", f"/v1/documents/{document_id}/passages:detect")
    ).json()
    second = asyncio.run(
        request(app, "POST", f"/v1/documents/{document_id}/passages:detect")
    ).json()

    assert second["passages"][0]["id"] == first["passages"][0]["id"]
    assert second["passages"][0]["label"] == "Passage 1"
    assert len(second["passages"]) == 1
    with app.state.database.session_factory() as session:
        rows = session.scalars(select(Passage).order_by(Passage.created_at)).all()
        assert [row.status for row in rows] == ["marked", "deleted"]


def test_redetection_keeps_a_retained_passage_label_stable_when_an_earlier_one_is_removed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document_id = _uploaded_document(tmp_path)
    results = [_result(), _result(include_first=False)]
    monkeypatch.setattr(passages_api, "detect_passages", lambda _path: results.pop(0))

    first = asyncio.run(
        request(app, "POST", f"/v1/documents/{document_id}/passages:detect")
    ).json()
    second = asyncio.run(
        request(app, "POST", f"/v1/documents/{document_id}/passages:detect")
    ).json()

    assert second["passages"][0]["id"] == first["passages"][1]["id"]
    assert second["passages"][0]["label"] == "Passage 2"


def test_redetection_collapses_preexisting_duplicate_detected_geometry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document_id = _uploaded_document(tmp_path)
    monkeypatch.setattr(
        passages_api,
        "detect_passages",
        lambda _path: _result(include_second=False),
    )
    original = asyncio.run(
        request(app, "POST", f"/v1/documents/{document_id}/passages:detect")
    ).json()["passages"][0]
    with app.state.database.session_factory() as session, session.begin():
        session.add(
            Passage(
                id="99999999-9999-4999-8999-999999999999",
                document_id=document_id,
                label="Duplicate",
                page_number=original["page_number"],
                rects=original["rects"],
                source=PassageSource.DETECTED.value,
                status=PassageStatus.MARKED.value,
                revision=1,
            )
        )

    response = asyncio.run(
        request(app, "POST", f"/v1/documents/{document_id}/passages:detect")
    )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["passages"]] == [original["id"]]
    with app.state.database.session_factory() as session:
        duplicate = session.get(Passage, "99999999-9999-4999-8999-999999999999")
        assert duplicate is not None
        assert duplicate.status == PassageStatus.DELETED.value


def test_manual_correction_replaces_an_overlapping_detected_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document_id = _uploaded_document(tmp_path)
    manual = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/documents/{document_id}/passages",
            headers={"Idempotency-Key": "manual-correction"},
            json={
                "label": "Corrected passage",
                "page_number": 1,
                "rects": [{"x": 0.09, "y": 0.19, "width": 0.22, "height": 0.04}],
            },
        )
    ).json()
    monkeypatch.setattr(
        passages_api,
        "detect_passages",
        lambda _path: _result(include_second=False),
    )

    response = asyncio.run(
        request(app, "POST", f"/v1/documents/{document_id}/passages:detect")
    )

    assert response.status_code == 200
    passages = response.json()["passages"]
    assert len(passages) == 1
    assert passages[0]["id"] == manual["id"]
    assert passages[0]["source"] == "manual"


def test_corrected_detected_passage_becomes_manual_and_survives_redetection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document_id = _uploaded_document(tmp_path)
    monkeypatch.setattr(
        passages_api,
        "detect_passages",
        lambda _path: _result(include_second=False),
    )
    detected = asyncio.run(
        request(app, "POST", f"/v1/documents/{document_id}/passages:detect")
    ).json()["passages"][0]

    corrected = asyncio.run(
        request(
            app,
            "PATCH",
            f"/v1/passages/{detected['id']}",
            json={
                "page_number": 1,
                "rects": [{"x": 0.09, "y": 0.19, "width": 0.22, "height": 0.04}],
            },
        )
    )
    redetected = asyncio.run(
        request(app, "POST", f"/v1/documents/{document_id}/passages:detect")
    )

    assert corrected.status_code == 200
    assert corrected.json()["source"] == "manual"
    passages = redetected.json()["passages"]
    assert len(passages) == 1
    assert passages[0]["id"] == detected["id"]
    assert passages[0]["source"] == "manual"
    assert passages[0]["rects"] == corrected.json()["rects"]


def test_deleted_detected_passage_remains_suppressed_after_redetection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document_id = _uploaded_document(tmp_path)
    monkeypatch.setattr(
        passages_api,
        "detect_passages",
        lambda _path: _result(include_second=False),
    )
    detected = asyncio.run(
        request(app, "POST", f"/v1/documents/{document_id}/passages:detect")
    ).json()["passages"][0]

    deleted = asyncio.run(request(app, "DELETE", f"/v1/passages/{detected['id']}"))
    redetected = asyncio.run(
        request(app, "POST", f"/v1/documents/{document_id}/passages:detect")
    )

    assert deleted.status_code == 200
    assert deleted.json()["source"] == "manual"
    assert redetected.status_code == 200
    assert redetected.json()["passages"] == []


def test_merged_detected_passages_remain_one_manual_correction_after_redetection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document_id = _uploaded_document(tmp_path)
    result = _same_page_result()
    monkeypatch.setattr(passages_api, "detect_passages", lambda _path: result)
    detected = asyncio.run(
        request(app, "POST", f"/v1/documents/{document_id}/passages:detect")
    ).json()["passages"]

    merged = asyncio.run(
        request(
            app,
            "POST",
            "/v1/passages:merge",
            json={"passage_ids": [item["id"] for item in detected]},
        )
    )
    redetected = asyncio.run(
        request(app, "POST", f"/v1/documents/{document_id}/passages:detect")
    )

    assert merged.status_code == 200
    assert merged.json()["passage"]["source"] == "manual"
    passages = redetected.json()["passages"]
    assert len(passages) == 1
    assert passages[0]["id"] == merged.json()["passage"]["id"]
    assert passages[0]["source"] == "manual"


def test_split_detected_passage_remains_manual_after_redetection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document_id = _uploaded_document(tmp_path)
    monkeypatch.setattr(
        passages_api,
        "detect_passages",
        lambda _path: _result(include_second=False),
    )
    detected = asyncio.run(
        request(app, "POST", f"/v1/documents/{document_id}/passages:detect")
    ).json()["passages"][0]
    parts = [
        {
            "label": "Passage 1a",
            "page_number": 1,
            "rects": [{"x": 0.1, "y": 0.2, "width": 0.1, "height": 0.03}],
        },
        {
            "label": "Passage 1b",
            "page_number": 1,
            "rects": [{"x": 0.2, "y": 0.2, "width": 0.1, "height": 0.03}],
        },
    ]

    split = asyncio.run(
        request(
            app,
            "POST",
            f"/v1/passages/{detected['id']}:split",
            json={"parts": parts},
        )
    )
    redetected = asyncio.run(
        request(app, "POST", f"/v1/documents/{document_id}/passages:detect")
    )

    assert split.status_code == 200
    assert [item["source"] for item in split.json()["passages"]] == [
        "manual",
        "manual",
    ]
    assert [item["id"] for item in redetected.json()["passages"]] == [
        item["id"] for item in split.json()["passages"]
    ]


def test_uncertain_detection_fails_closed_without_persisting_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app, document_id = _uploaded_document(tmp_path)
    uncertain = _result().model_copy(
        update={
            "coverage_confident": False,
            "uncertain_pages": [2],
            "warnings": ["page 2: insufficient_visible_content"],
        }
    )
    monkeypatch.setattr(passages_api, "detect_passages", lambda _path: uncertain)

    response = asyncio.run(
        request(app, "POST", f"/v1/documents/{document_id}/passages:detect")
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "mask_coverage_uncertain",
        "message": "Automatic detection could not confirm every page",
        "uncertain_pages": [2],
        "warnings": ["page 2: insufficient_visible_content"],
    }
    with app.state.database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(Passage)) == 0


def _uploaded_document(tmp_path: Path):
    app = create_app(
        Settings(
            database_url=f"sqlite+pysqlite:///{tmp_path / 'detection.sqlite3'}",
            document_storage_root=tmp_path / "stored-documents",
        )
    )
    Base.metadata.create_all(app.state.database.engine)
    fixture = Path(__file__).parents[1] / "fixtures" / "pdfs" / "visible-layout.pdf"
    response = asyncio.run(
        request(
            app,
            "POST",
            "/v1/documents",
            files={"file": (fixture.name, fixture.read_bytes(), "application/pdf")},
        )
    )
    assert response.status_code == 201
    return app, response.json()["id"]


def _result(
    *,
    include_first: bool = True,
    include_second: bool = True,
) -> DetectionResult:
    passages = []
    if include_first:
        passages.append(
            DetectedPassage(
                page_number=1,
                rects=[PassageRect(x=0.1, y=0.2, width=0.2, height=0.03)],
                kinds=["black"],
                source_mask_ids=["mask-1"],
                confidence=1.0,
            )
        )
    if include_second:
        passages.append(
            DetectedPassage(
                page_number=2,
                rects=[PassageRect(x=0.1, y=0.4, width=0.2, height=0.03)],
                kinds=["white"],
                source_mask_ids=["mask-2"],
                confidence=0.8,
            )
        )
    return DetectionResult(
        passages=passages,
        pages_scanned=2,
        coverage_confident=True,
        uncertain_pages=[],
        warnings=[],
    )


def _same_page_result() -> DetectionResult:
    return DetectionResult(
        passages=[
            DetectedPassage(
                page_number=1,
                rects=[PassageRect(x=0.1, y=0.2, width=0.2, height=0.03)],
                kinds=["black"],
                source_mask_ids=["mask-1"],
                confidence=1.0,
            ),
            DetectedPassage(
                page_number=1,
                rects=[PassageRect(x=0.1, y=0.4, width=0.2, height=0.03)],
                kinds=["white"],
                source_mask_ids=["mask-2"],
                confidence=0.8,
            ),
        ],
        pages_scanned=2,
        coverage_confident=True,
        uncertain_pages=[],
        warnings=[],
    )
