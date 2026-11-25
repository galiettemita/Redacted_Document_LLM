import asyncio
import hashlib
import json
import platform as runtime_platform
import struct
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

import redacted_backend.api.readiness as readiness_api
from redacted_backend.api.readiness import (
    resolve_best_available_corpus_readiness,
    resolve_corpus_readiness,
    resolve_staged_bm25_corpus_readiness,
    resolve_worker_readiness,
)
from redacted_backend.config import Settings
from redacted_backend.corpus.best_available import (
    BEST_AVAILABLE_BM25_INDEX_NAME,
    BestAvailableCorpusManifest,
    BestAvailableRetrievalContract,
)
from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.corpus.manifest import CorpusManifest
from redacted_backend.domain.readiness import (
    verify_best_available_corpus_readiness,
    verify_corpus_readiness,
    verify_staged_bm25_readiness,
)
from redacted_backend.domain.analysis import PredictionBasis
from redacted_backend.domain.readiness import ComponentReadiness
from redacted_backend.main import create_app
from redacted_backend.retrieval.bm25 import index_definition_hash
from redacted_backend.retrieval.dense_profile import (
    DenseArtifactProfile,
    dense_semantics_hash,
    faiss_layout_hash,
)
from redacted_backend.retrieval.types import (
    Bm25IndexManifest,
    CompressedDenseShardManifest,
    CompressedShardedDenseIndexManifest,
    DenseFaissVerification,
    DenseLatencyReport,
    DenseRecallReport,
    DenseIndexManifest,
)
from redacted_backend.scientific_config import ScientificConfig
from redacted_backend.runtime_identity import WorkerHeartbeat, WorkerRuntimeIdentity


def _test_worker_identity() -> WorkerRuntimeIdentity:
    return WorkerRuntimeIdentity(
        worker_build_id="a" * 64,
        serving_host_id="b" * 64,
        platform=runtime_platform.platform(),
        machine=runtime_platform.machine(),
    )


def test_readiness_never_equates_http_health_with_prediction_readiness() -> None:
    settings = Settings(
        database_url=None,
        redis_url=None,
        corpus_manifest_path=None,
        openai_api_key=None,
    )

    async def request_readiness():
        transport = ASGITransport(app=create_app(settings))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/v1/readiness")

    response = asyncio.run(request_readiness())

    assert response.status_code == 200
    body = response.json()
    assert body["http"] == {"ready": True, "detail": "ready"}
    assert body["database"]["ready"] is False
    assert body["worker"]["ready"] is False
    assert body["corpus"]["ready"] is False
    assert body["model"]["ready"] is False
    assert body["prediction_ready"] is False
    assert set(body) == {
        "http",
        "database",
        "worker",
        "corpus",
        "model",
        "prediction_ready",
        "prediction_bases",
        "strongest_ready_basis",
    }
    assert body["strongest_ready_basis"] is None
    assert set(body["prediction_bases"]) == {
        "model_only_v1",
        "best_available_foiarchive_bm25_v1",
        "best_available_foiarchive_v1",
        "paper_exact_v1",
    }


def test_http_health_does_not_probe_prediction_dependencies(monkeypatch) -> None:
    def unexpected_probe(_settings: Settings):
        raise AssertionError("liveness must not probe prediction dependencies")

    monkeypatch.setattr(
        "redacted_backend.api.readiness.resolve_corpus_readiness",
        unexpected_probe,
    )

    async def request_health():
        transport = ASGITransport(app=create_app(Settings()))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/v1/health")

    response = asyncio.run(request_health())

    assert response.status_code == 200
    assert response.json() == {"ready": True, "detail": "ready"}


def test_database_readiness_probes_the_configured_database(tmp_path: Path) -> None:
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'readiness.db'}",
        document_storage_root=tmp_path / "documents",
    )

    async def request_readiness():
        transport = ASGITransport(app=create_app(settings))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/v1/readiness")

    body = asyncio.run(request_readiness()).json()

    assert body["database"] == {"ready": True, "detail": "ready"}
    assert body["prediction_ready"] is False


def test_worker_readiness_requires_a_live_expiring_heartbeat() -> None:
    class FakeRedis:
        def __init__(self, value: bytes | None) -> None:
            self.value = value

        def get(self, _key: str) -> bytes | None:
            return self.value

    settings = Settings(redis_url="redis://worker.test/0")
    heartbeat = WorkerHeartbeat(identity=_test_worker_identity()).model_dump_json().encode()

    ready = resolve_worker_readiness(
        settings,
        client_factory=lambda _url: FakeRedis(heartbeat),
    )
    stale = resolve_worker_readiness(
        settings,
        client_factory=lambda _url: FakeRedis(None),
    )

    assert ready.model_dump() == {"ready": True, "detail": "heartbeat current"}
    assert stale.model_dump() == {
        "ready": False,
        "detail": "heartbeat missing or expired",
    }
    legacy = resolve_worker_readiness(
        settings,
        client_factory=lambda _url: FakeRedis(b"ready"),
    )
    assert legacy.ready is False


def test_model_readiness_requires_access_and_explicit_paid_call_approval() -> None:
    async def read(settings: Settings):
        transport = ASGITransport(app=create_app(settings))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return (await client.get("/v1/readiness")).json()

    unapproved = asyncio.run(
        read(Settings(openai_api_key="secret", allow_paid_model_calls=False))
    )
    approved = asyncio.run(
        read(Settings(openai_api_key="secret", allow_paid_model_calls=True))
    )

    assert unapproved["model"] == {
        "ready": False,
        "detail": "configured; paid calls not approved",
    }
    assert approved["model"] == {
        "ready": True,
        "detail": "configured and paid calls approved",
    }

    empty = asyncio.run(read(Settings(openai_api_key="", allow_paid_model_calls=True)))
    assert empty["model"] == {"ready": False, "detail": "not configured"}


def test_model_only_basis_is_operational_without_corpus_but_separately_approved() -> None:
    resolve = getattr(readiness_api, "resolve_prediction_basis_readiness", None)
    select = getattr(readiness_api, "select_prediction_basis", None)
    assert resolve is not None
    assert select is not None
    ready = ComponentReadiness(ready=True, detail="ready")
    corpus_unavailable = ComponentReadiness(ready=False, detail="not configured")
    settings = Settings(
        openai_api_key="secret",
        allow_paid_model_calls=True,
        allow_model_only_calls=True,
        allow_best_available_calls=True,
        allow_paper_exact_calls=True,
        provider_spend_cap_usd="10.00",
    )

    bases = resolve(
        settings,
        database=ready,
        worker=ready,
        corpus=corpus_unavailable,
        best_available_corpus=corpus_unavailable,
    )

    assert bases[PredictionBasis.MODEL_ONLY_V1].model_dump() == {
        "operational": True,
        "approved": True,
        "ready": True,
        "detail": "operational and approved",
    }
    assert bases[PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1].model_dump() == {
        "operational": False,
        "approved": True,
        "ready": False,
        "detail": "corpus: not configured",
    }
    assert bases[PredictionBasis.PAPER_EXACT_V1].model_dump() == {
        "operational": False,
        "approved": True,
        "ready": False,
        "detail": "corpus: not configured",
    }
    assert select(bases) == PredictionBasis.MODEL_ONLY_V1


def test_model_only_basis_requires_master_and_its_own_spending_approval() -> None:
    resolve = getattr(readiness_api, "resolve_prediction_basis_readiness", None)
    assert resolve is not None
    ready = ComponentReadiness(ready=True, detail="ready")
    corpus_unavailable = ComponentReadiness(ready=False, detail="not configured")

    per_basis_unapproved = resolve(
        Settings(
            openai_api_key="secret",
            allow_paid_model_calls=True,
            allow_model_only_calls=False,
            provider_spend_cap_usd="10.00",
        ),
        database=ready,
        worker=ready,
        corpus=corpus_unavailable,
        best_available_corpus=corpus_unavailable,
    )[PredictionBasis.MODEL_ONLY_V1]
    master_unapproved = resolve(
        Settings(
            openai_api_key="secret",
            allow_paid_model_calls=False,
            allow_model_only_calls=True,
            provider_spend_cap_usd="10.00",
        ),
        database=ready,
        worker=ready,
        corpus=corpus_unavailable,
        best_available_corpus=corpus_unavailable,
    )[PredictionBasis.MODEL_ONLY_V1]

    assert per_basis_unapproved.model_dump() == {
        "operational": True,
        "approved": False,
        "ready": False,
        "detail": "operational; spending not approved",
    }
    assert master_unapproved == per_basis_unapproved


def test_best_available_basis_is_operational_without_contaminating_exact() -> None:
    ready = ComponentReadiness(ready=True, detail="ready")
    unavailable = ComponentReadiness(ready=False, detail="not configured")
    settings = Settings(
        openai_api_key="secret",
        allow_paid_model_calls=True,
        allow_best_available_calls=True,
        allow_paper_exact_calls=True,
        provider_spend_cap_usd="10.00",
    )

    bases = readiness_api.resolve_prediction_basis_readiness(
        settings,
        database=ready,
        worker=ready,
        corpus=unavailable,
        best_available_corpus=ready,
    )

    assert bases[PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1].ready is True
    assert bases[PredictionBasis.PAPER_EXACT_V1].ready is False
    assert readiness_api.select_prediction_basis(bases) == (
        PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1
    )


def test_staged_basis_requires_its_own_approval_cap_and_priority() -> None:
    ready = ComponentReadiness(ready=True, detail="ready")
    unavailable = ComponentReadiness(ready=False, detail="not configured")
    settings = Settings(
        openai_api_key="secret",
        allow_paid_model_calls=True,
        allow_model_only_calls=True,
        allow_staged_bm25_calls=True,
        allow_best_available_calls=True,
        allow_paper_exact_calls=True,
        staged_bm25_spend_cap_usd="10.00",
    )

    all_ready = readiness_api.resolve_prediction_basis_readiness(
        settings,
        database=ready,
        worker=ready,
        corpus=ready,
        staged_bm25_corpus=ready,
        best_available_corpus=ready,
    )
    assert readiness_api.select_prediction_basis(all_ready) == (
        PredictionBasis.PAPER_EXACT_V1
    )

    without_exact = readiness_api.resolve_prediction_basis_readiness(
        settings,
        database=ready,
        worker=ready,
        corpus=unavailable,
        staged_bm25_corpus=ready,
        best_available_corpus=ready,
    )
    assert readiness_api.select_prediction_basis(without_exact) == (
        PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1
    )

    staged_only = readiness_api.resolve_prediction_basis_readiness(
        settings,
        database=ready,
        worker=ready,
        corpus=unavailable,
        staged_bm25_corpus=ready,
        best_available_corpus=unavailable,
    )
    assert staged_only[
        PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_BM25_V1
    ].model_dump() == {
        "operational": True,
        "approved": True,
        "ready": True,
        "detail": "operational and approved",
    }
    assert readiness_api.select_prediction_basis(staged_only) == (
        PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_BM25_V1
    )

    missing_cap = readiness_api.resolve_prediction_basis_readiness(
        settings.model_copy(update={"staged_bm25_spend_cap_usd": None}),
        database=ready,
        worker=ready,
        corpus=unavailable,
        staged_bm25_corpus=ready,
        best_available_corpus=unavailable,
    )[PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_BM25_V1]
    assert missing_cap.operational is False
    assert missing_cap.detail == "spend cap: not configured"

    unapproved = readiness_api.resolve_prediction_basis_readiness(
        settings.model_copy(update={"allow_staged_bm25_calls": False}),
        database=ready,
        worker=ready,
        corpus=unavailable,
        staged_bm25_corpus=ready,
        best_available_corpus=unavailable,
    )[PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_BM25_V1]
    assert unapproved.operational is True
    assert unapproved.approved is False
    assert unapproved.ready is False


def test_every_prediction_basis_fails_closed_without_a_provider_spend_cap() -> None:
    ready = ComponentReadiness(ready=True, detail="ready")
    settings = Settings(
        openai_api_key="secret",
        allow_paid_model_calls=True,
        allow_model_only_calls=True,
        allow_staged_bm25_calls=True,
        allow_best_available_calls=True,
        allow_paper_exact_calls=True,
        provider_spend_cap_usd=None,
        staged_bm25_spend_cap_usd=None,
    )

    bases = readiness_api.resolve_prediction_basis_readiness(
        settings,
        database=ready,
        worker=ready,
        corpus=ready,
        staged_bm25_corpus=ready,
        best_available_corpus=ready,
    )

    assert all(not status.operational for status in bases.values())
    assert all(status.detail == "spend cap: not configured" for status in bases.values())


def test_staged_bm25_readiness_needs_only_verified_corpus_and_bm25(
    tmp_path: Path,
) -> None:
    settings, corpus, bm25, dense = _compatible_best_available_settings(tmp_path)
    settings = settings.model_copy(
        update={
            "best_available_dense_manifest_path": None,
            "best_available_dense_artifact_directory": None,
            "best_available_chunks_path": None,
            "best_available_recall_queries_path": None,
            "best_available_latency_report_path": None,
        }
    )
    probes: list[tuple[str, str]] = []

    result = resolve_staged_bm25_corpus_readiness(
        settings,
        index_counter=lambda _url, _index: bm25.chunk_count,
        search_probe=lambda url, index: probes.append((url, index)),
        contract=_best_available_test_contract(corpus, dense),
    )

    assert verify_staged_bm25_readiness(
        corpus,
        bm25,
        ScientificConfig(),
        contract=_best_available_test_contract(corpus, dense),
    ).ready is True
    assert result.model_dump() == {
        "ready": True,
        "detail": "best-available corpus and BM25 index operational",
    }
    assert probes == [(settings.opensearch_url, bm25.index_name)]


def test_staged_bm25_readiness_fails_closed_on_paths_count_probe_or_identity(
    tmp_path: Path,
) -> None:
    settings, corpus, bm25, dense = _compatible_best_available_settings(tmp_path)
    contract = _best_available_test_contract(corpus, dense)

    missing = resolve_staged_bm25_corpus_readiness(
        settings.model_copy(update={"best_available_bm25_manifest_path": None}),
        contract=contract,
    )
    assert missing.model_dump() == {"ready": False, "detail": "not configured"}

    mismatch = resolve_staged_bm25_corpus_readiness(
        settings,
        index_counter=lambda _url, _index: bm25.chunk_count + 1,
        search_probe=lambda _url, _index: None,
        contract=contract,
    )
    assert mismatch.detail == "BM25 live index count mismatch"

    def failed_probe(_url: str, _index: str) -> None:
        raise RuntimeError("private search detail")

    probe_failure = resolve_staged_bm25_corpus_readiness(
        settings,
        index_counter=lambda _url, _index: bm25.chunk_count,
        search_probe=failed_probe,
        contract=contract,
    )
    assert probe_failure.detail == "BM25 search probe failed"

    assert settings.best_available_bm25_manifest_path is not None
    settings.best_available_bm25_manifest_path.write_text(
        bm25.model_copy(update={"index_name": "wrong-index"}).model_dump_json(),
        encoding="utf-8",
    )
    identity = resolve_staged_bm25_corpus_readiness(
        settings,
        index_counter=lambda _url, _index: bm25.chunk_count,
        search_probe=lambda _url, _index: None,
        contract=contract,
    )
    assert identity.detail == "best-available BM25 profile mismatch"


def test_corpus_readiness_requires_dense_artifact_files(
    tmp_path: Path,
) -> None:
    settings, _ = _compatible_readiness_settings(tmp_path)

    readiness = resolve_corpus_readiness(
        settings,
        index_counter=lambda _url, _index: 1,
    )

    assert readiness.model_dump() == {
        "ready": False,
        "detail": "dense artifacts not available",
    }


def test_corpus_readiness_requires_a_configured_retrieval_service(
    tmp_path: Path,
) -> None:
    settings, _ = _compatible_readiness_settings(tmp_path, create_dense=True)

    readiness = resolve_corpus_readiness(
        settings,
        index_counter=lambda _url, _index: 1,
    )

    assert readiness.model_dump() == {
        "ready": False,
        "detail": "retrieval service not configured",
    }


def test_corpus_readiness_fails_closed_when_the_live_index_probe_fails(
    tmp_path: Path,
) -> None:
    settings, _ = _compatible_readiness_settings(
        tmp_path,
        create_dense=True,
        opensearch_url="http://search.test:9200",
    )

    def unavailable(_url: str, _index: str) -> int:
        raise RuntimeError("credential-or-host-detail")

    readiness = resolve_corpus_readiness(settings, index_counter=unavailable)

    assert readiness.model_dump() == {
        "ready": False,
        "detail": "retrieval index probe failed",
    }


def test_corpus_readiness_rejects_a_live_index_count_mismatch(
    tmp_path: Path,
) -> None:
    settings, _ = _compatible_readiness_settings(
        tmp_path,
        create_dense=True,
        opensearch_url="http://search.test:9200",
    )

    readiness = resolve_corpus_readiness(
        settings,
        index_counter=lambda _url, _index: 2,
    )

    assert readiness.model_dump() == {
        "ready": False,
        "detail": "BM25 live index count mismatch",
    }


def test_corpus_readiness_requires_matching_manifests_artifacts_and_live_index(
    tmp_path: Path,
) -> None:
    settings, index = _compatible_readiness_settings(
        tmp_path,
        create_dense=True,
        opensearch_url="http://search.test:9200",
    )

    readiness = resolve_corpus_readiness(
        settings,
        index_counter=lambda url, name: (
            index.chunk_count
            if (url, name) == ("http://search.test:9200", index.index_name)
            else -1
        ),
    )

    assert readiness.model_dump() == {
        "ready": True,
        "detail": "corpus and retrieval indexes operational",
    }


def test_best_available_readiness_verifies_separate_shards_queries_and_live_index(
    tmp_path: Path,
) -> None:
    settings, corpus, bm25, dense = _compatible_best_available_settings(tmp_path)
    contract = _best_available_test_contract(corpus, dense)

    readiness = resolve_best_available_corpus_readiness(
        settings,
        index_counter=lambda url, name: (
            bm25.chunk_count
            if (url, name) == ("http://search.test:9200", bm25.index_name)
            else -1
        ),
        contract=contract,
        worker_identity=_test_worker_identity(),
    )

    assert readiness.model_dump() == {
        "ready": True,
        "detail": "best-available corpus and retrieval indexes operational",
    }
    assert verify_best_available_corpus_readiness(
        corpus,
        bm25,
        dense,
        ScientificConfig(),
        contract=contract,
    ).ready is True
    assert verify_best_available_corpus_readiness(
        corpus,
        bm25,
        dense,
        ScientificConfig(),
    ).model_dump() == {
        "ready": False,
        "detail": "best-available production corpus identity mismatch",
    }
    ready = ComponentReadiness(ready=True, detail="ready")
    unavailable = ComponentReadiness(ready=False, detail="not configured")
    basis = readiness_api.resolve_prediction_basis_readiness(
        settings.model_copy(
            update={
                "openai_api_key": "configured",
                "provider_spend_cap_usd": "10.00",
            }
        ),
        database=ready,
        worker=ready,
        corpus=unavailable,
        best_available_corpus=readiness,
    )[PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_V1]
    assert basis.model_dump() == {
        "operational": True,
        "approved": False,
        "ready": False,
        "detail": "operational; spending not approved",
    }


def test_best_available_readiness_rejects_latency_from_another_platform(
    tmp_path: Path,
) -> None:
    settings, corpus, _bm25, dense = _compatible_best_available_settings(tmp_path)
    latency_path = settings.best_available_latency_report_path
    assert latency_path is not None
    latency = DenseLatencyReport.model_validate_json(
        latency_path.read_text(encoding="utf-8")
    )
    latency_path.write_text(
        latency.model_copy(update={"platform": "Linux-cloud-host"}).model_dump_json(),
        encoding="utf-8",
    )

    result = resolve_best_available_corpus_readiness(
        settings,
        index_counter=lambda *_args: pytest.fail("live index should not be probed"),
        contract=_best_available_test_contract(corpus, dense),
        worker_identity=_test_worker_identity(),
    )

    assert result.ready is False
    assert result.detail == "dense latency report invalid"


@pytest.mark.parametrize(
    "identity_update",
    [
        {"worker_build_id": "c" * 64},
        {"serving_host_id": "c" * 64},
    ],
)
def test_best_available_readiness_rejects_latency_from_another_worker(
    tmp_path: Path,
    identity_update: dict[str, str],
) -> None:
    settings, corpus, _bm25, dense = _compatible_best_available_settings(tmp_path)

    result = resolve_best_available_corpus_readiness(
        settings,
        index_counter=lambda *_args: pytest.fail("live index should not be probed"),
        contract=_best_available_test_contract(corpus, dense),
        worker_identity=_test_worker_identity().model_copy(update=identity_update),
    )

    assert result.model_dump() == {
        "ready": False,
        "detail": "dense latency report invalid",
    }


def test_best_available_readiness_fails_closed_on_corruption_or_live_count(
    tmp_path: Path,
) -> None:
    settings, _corpus, bm25, dense = _compatible_best_available_settings(tmp_path)
    contract = _best_available_test_contract(_corpus, dense)
    assert resolve_best_available_corpus_readiness(
        settings,
        index_counter=lambda _url, _name: bm25.chunk_count,
        contract=contract,
        worker_identity=_test_worker_identity(),
    ).ready is True
    shard_root = settings.best_available_dense_artifact_directory
    assert shard_root is not None
    (shard_root / dense.shards[0].path / "candidate.faiss").write_bytes(b"tampered")

    corrupt = resolve_best_available_corpus_readiness(
        settings,
        index_counter=lambda _url, _name: bm25.chunk_count,
        contract=contract,
        worker_identity=_test_worker_identity(),
    )
    assert corrupt.model_dump() == {
        "ready": False,
        "detail": "dense shard artifacts invalid",
    }

    settings, _corpus, bm25, _dense = _compatible_best_available_settings(
        tmp_path / "count"
    )
    contract = _best_available_test_contract(_corpus, _dense)
    mismatch = resolve_best_available_corpus_readiness(
        settings,
        index_counter=lambda _url, _name: bm25.chunk_count + 1,
        contract=contract,
        worker_identity=_test_worker_identity(),
    )
    assert mismatch.model_dump() == {
        "ready": False,
        "detail": "BM25 live index count mismatch",
    }


@pytest.mark.parametrize(
    "case",
    ["recall_report", "candidate_header", "shard_manifest"],
)
def test_best_available_readiness_parses_scientific_artifacts(
    tmp_path: Path,
    case: str,
) -> None:
    settings, corpus, bm25, dense = _compatible_best_available_settings(tmp_path)
    dense_root = settings.best_available_dense_artifact_directory
    assert dense_root is not None
    shard = dense.shards[0]
    if case == "recall_report":
        path = dense_root / dense.recall_report_path
        previous_size = path.stat().st_size
        path.write_text("{}\n", encoding="utf-8")
        dense = dense.model_copy(
            update={
                "recall_report_size_bytes": path.stat().st_size,
                "recall_report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "total_artifact_bytes": (
                    dense.total_artifact_bytes - previous_size + path.stat().st_size
                ),
            }
        )
    elif case == "candidate_header":
        path = dense_root / shard.path / "candidate.faiss"
        previous_size = path.stat().st_size
        path.write_bytes(b"not-a-faiss-wrapper")
        shard = shard.model_copy(
            update={
                "candidate_index_size_bytes": path.stat().st_size,
                "candidate_index_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        dense = dense.model_copy(
            update={
                "shards": [shard],
                "total_artifact_bytes": (
                    dense.total_artifact_bytes - previous_size + path.stat().st_size
                ),
            }
        )
        (dense_root / shard.path / "manifest.json").write_text(
            shard.model_dump_json() + "\n",
            encoding="utf-8",
        )
    else:
        (dense_root / shard.path / "manifest.json").write_text(
            shard.model_copy(
                update={"candidate_index_sha256": "0" * 64}
            ).model_dump_json()
            + "\n",
            encoding="utf-8",
        )
    _rewrite_dense_fixture(settings, dense)
    contract = _best_available_test_contract(corpus, dense)

    readiness = resolve_best_available_corpus_readiness(
        settings,
        index_counter=lambda _url, _name: bm25.chunk_count,
        contract=contract,
        worker_identity=_test_worker_identity(),
    )

    assert readiness.model_dump() == {
        "ready": False,
        "detail": "dense shard artifacts invalid",
    }


@pytest.mark.parametrize(
    ("case", "detail"),
    [
        ("unsupported_profile", "unsupported dense artifact profile"),
        ("profile_hash", "dense artifact profile hash mismatch"),
        ("semantics", "dense semantic manifest mismatch"),
        ("chunks", "dense chunk checksum mismatch"),
        ("offset_size", "dense offset size mismatch"),
        ("shard_range", "dense shard range mismatch"),
        ("candidate_template", "dense candidate template mismatch"),
        ("recall_count", "dense recall query identity mismatch"),
        ("recall_value", "dense recall@300 is below 0.9800"),
    ],
)
def test_best_available_manifest_mismatches_fail_closed(
    tmp_path: Path,
    case: str,
    detail: str,
) -> None:
    _settings, corpus, bm25, dense = _compatible_best_available_settings(
        tmp_path / case
    )
    contract = _best_available_test_contract(corpus, dense)
    if case == "unsupported_profile":
        unsupported = dense.artifact_profile.model_copy(update={"nprobe": 512})
        dense = dense.model_copy(
            update={
                "artifact_profile": unsupported,
                "artifact_profile_hash": unsupported.manifest_hash(),
            }
        )
    elif case == "profile_hash":
        dense = dense.model_copy(update={"artifact_profile_hash": "0" * 64})
    elif case == "semantics":
        dense = dense.model_copy(update={"dense_semantics_hash": "0" * 64})
    elif case == "chunks":
        dense = dense.model_copy(update={"chunks_sha256": "0" * 64})
    elif case == "offset_size":
        dense = dense.model_copy(update={"chunk_offsets_size_bytes": 16})
    elif case == "shard_range":
        bad_shard = dense.shards[0].model_copy(update={"start_row": 1})
        dense = dense.model_copy(update={"shards": [bad_shard]})
    elif case == "candidate_template":
        bad_shard = dense.shards[0].model_copy(
            update={"candidate_template_sha256": "0" * 64}
        )
        dense = dense.model_copy(update={"shards": [bad_shard]})
    elif case == "recall_count":
        dense = dense.model_copy(update={"recall_query_count": 99})
    elif case == "recall_value":
        dense = dense.model_copy(update={"recall_at_300": 0.9799})

    assert verify_best_available_corpus_readiness(
        corpus,
        bm25,
        dense,
        ScientificConfig(),
        contract=contract,
    ).model_dump() == {"ready": False, "detail": detail}


@pytest.mark.parametrize(
    "case",
    [
        "offset_checksum",
        "shard_size",
        "embedding_checksum",
        "missing_embedding",
        "shard_checksum",
        "template_checksum",
        "query_hash",
        "latency_manifest",
        "latency_query",
        "latency_count",
        "latency_maximum",
        "missing_chunks",
    ],
)
def test_best_available_external_mismatches_fail_before_live_bm25_probe(
    tmp_path: Path,
    case: str,
) -> None:
    settings, _corpus, _bm25, dense = _compatible_best_available_settings(
        tmp_path / case
    )
    contract = _best_available_test_contract(_corpus, dense)
    dense_root = settings.best_available_dense_artifact_directory
    latency_path = settings.best_available_latency_report_path
    queries_path = settings.best_available_recall_queries_path
    assert dense_root is not None
    assert latency_path is not None
    assert queries_path is not None
    if case == "offset_checksum":
        path = dense_root / dense.chunk_offsets_path
        path.write_bytes(b"1" * path.stat().st_size)
    elif case == "shard_size":
        path = dense_root / dense.shards[0].path / "embeddings.i8"
        path.write_bytes(path.read_bytes() + b"x")
    elif case == "embedding_checksum":
        path = dense_root / dense.shards[0].path / "embeddings.i8"
        path.write_bytes(b"\2" * path.stat().st_size)
    elif case == "missing_embedding":
        (dense_root / dense.shards[0].path / "embeddings.i8").unlink()
    elif case == "shard_checksum":
        path = dense_root / dense.shards[0].path / "candidate.faiss"
        path.write_bytes(b"x" * path.stat().st_size)
    elif case == "template_checksum":
        path = dense_root / dense.candidate_template_path
        path.write_bytes(b"x" * path.stat().st_size)
    elif case == "query_hash":
        queries_path.write_text("changed\n", encoding="utf-8")
    elif case.startswith("latency_"):
        latency = DenseLatencyReport.model_validate_json(
            latency_path.read_text(encoding="utf-8")
        )
        updates = {
            "latency_manifest": {"dense_manifest_hash": "0" * 64},
            "latency_query": {"recall_queries_sha256": "0" * 64},
            "latency_count": {"query_count": 99},
            "latency_maximum": {"maximum_allowed_seconds": 4.0},
        }[case]
        latency_path.write_text(
            latency.model_copy(update=updates).model_dump_json(),
            encoding="utf-8",
        )
    elif case == "missing_chunks":
        settings = settings.model_copy(update={"best_available_chunks_path": None})
    probe_calls = 0

    def count(_url: str, _index: str) -> int:
        nonlocal probe_calls
        probe_calls += 1
        return 1

    readiness = resolve_best_available_corpus_readiness(
        settings,
        index_counter=count,
        contract=contract,
        worker_identity=_test_worker_identity(),
    )

    assert readiness.ready is False
    assert probe_calls == 0


def test_best_available_readiness_rejects_old_float16_manifest_before_probe(
    tmp_path: Path,
) -> None:
    settings, corpus, _bm25, dense = _compatible_best_available_settings(tmp_path)
    dense_path = settings.best_available_dense_manifest_path
    assert dense_path is not None
    value = dense.model_dump(mode="json")
    value.update(
        {
            "schema_version": "sharded-dense-index-manifest-v2",
            "embedding_file": "embeddings.f16",
            "dtype": "float16",
        }
    )
    dense_path.write_text(json.dumps(value), encoding="utf-8")
    probe_calls = 0

    def count(_url: str, _index: str) -> int:
        nonlocal probe_calls
        probe_calls += 1
        return 1

    readiness = resolve_best_available_corpus_readiness(
        settings,
        index_counter=count,
        contract=_best_available_test_contract(corpus, dense),
        worker_identity=_test_worker_identity(),
    )

    assert readiness.model_dump() == {"ready": False, "detail": "manifest invalid"}
    assert probe_calls == 0


def _compatible_readiness_settings(
    tmp_path: Path,
    *,
    create_dense: bool = False,
    opensearch_url: str | None = None,
) -> tuple[Settings, Bm25IndexManifest]:
    scientific = ScientificConfig()
    corpus = CorpusManifest(
        schema_version="historylab-normalized-v1",
        source_manifest_hash="1" * 64,
        coverage_report_hash="4" * 64,
        exclusion_hash="2" * 64,
        scientific_hash=scientific.manifest_hash(),
        chunker_version=scientific.corpus_chunker_version,
        chunk_words=scientific.chunk_words,
        chunk_overlap=scientific.chunk_overlap,
        document_count=1,
        chunk_count=1,
        corpus_counts={"cfpf": 1},
        exclusion_counts={},
        coverage_ready=True,
        chunks_sha256="3" * 64,
    )
    index = Bm25IndexManifest(
        index_name=scientific.bm25_index_name,
        opensearch_version=scientific.opensearch_version,
        corpus_manifest_hash=corpus.manifest_hash(),
        scientific_hash=scientific.manifest_hash(),
        stopwords_hash=scientific.context_stopwords_hash(),
        index_definition_hash=index_definition_hash(scientific),
        document_count=1,
        chunk_count=1,
    )
    dense = _dense_manifest(scientific, corpus)
    corpus_path = tmp_path / "corpus.json"
    index_path = tmp_path / "bm25.json"
    dense_path = tmp_path / "dense.json"
    dense_directory = tmp_path / "dense"
    corpus_path.write_text(json.dumps(corpus.model_dump(mode="json")), encoding="utf-8")
    index_path.write_text(json.dumps(index.model_dump(mode="json")), encoding="utf-8")
    dense_path.write_text(json.dumps(dense.model_dump(mode="json")), encoding="utf-8")
    if create_dense:
        dense_directory.mkdir()
        for name in ("embeddings.f16", "metadata.jsonl", "candidate.faiss"):
            (dense_directory / name).write_bytes(b"fixture")
    settings = Settings(
        database_url=None,
        redis_url=None,
        corpus_manifest_path=corpus_path,
        bm25_manifest_path=index_path,
        dense_manifest_path=dense_path,
        dense_artifact_directory=dense_directory,
        opensearch_url=opensearch_url,
        openai_api_key=None,
    )
    return settings, index


def test_corpus_readiness_rejects_an_incompatible_bm25_definition_or_count() -> None:
    scientific = ScientificConfig()
    corpus = CorpusManifest(
        schema_version="historylab-normalized-v1",
        source_manifest_hash="1" * 64,
        coverage_report_hash="4" * 64,
        exclusion_hash="2" * 64,
        scientific_hash=scientific.manifest_hash(),
        chunker_version=scientific.corpus_chunker_version,
        chunk_words=scientific.chunk_words,
        chunk_overlap=scientific.chunk_overlap,
        document_count=1,
        chunk_count=1,
        corpus_counts={"cfpf": 1},
        exclusion_counts={},
        coverage_ready=True,
        chunks_sha256="3" * 64,
    )
    compatible = Bm25IndexManifest(
        index_name=scientific.bm25_index_name,
        opensearch_version=scientific.opensearch_version,
        corpus_manifest_hash=corpus.manifest_hash(),
        scientific_hash=scientific.manifest_hash(),
        stopwords_hash=scientific.context_stopwords_hash(),
        index_definition_hash=index_definition_hash(scientific),
        document_count=1,
        chunk_count=1,
    )
    dense = _dense_manifest(scientific, corpus)

    wrong_definition = compatible.model_copy(update={"index_definition_hash": "0" * 64})
    wrong_count = compatible.model_copy(update={"chunk_count": 2})

    assert verify_corpus_readiness(
        corpus, wrong_definition, dense, scientific
    ).model_dump() == {
        "ready": False,
        "detail": "BM25 index definition mismatch",
    }
    assert verify_corpus_readiness(
        corpus, wrong_count, dense, scientific
    ).model_dump() == {
        "ready": False,
        "detail": "BM25 corpus count mismatch",
    }


def test_corpus_readiness_rejects_dense_revision_and_recall_mismatches() -> None:
    scientific = ScientificConfig()
    corpus = CorpusManifest(
        schema_version="historylab-normalized-v1",
        source_manifest_hash="1" * 64,
        coverage_report_hash="4" * 64,
        exclusion_hash="2" * 64,
        scientific_hash=scientific.manifest_hash(),
        chunker_version=scientific.corpus_chunker_version,
        chunk_words=scientific.chunk_words,
        chunk_overlap=scientific.chunk_overlap,
        document_count=1,
        chunk_count=1,
        corpus_counts={"cfpf": 1},
        exclusion_counts={},
        coverage_ready=True,
        chunks_sha256="3" * 64,
    )
    bm25 = Bm25IndexManifest(
        index_name=scientific.bm25_index_name,
        opensearch_version=scientific.opensearch_version,
        corpus_manifest_hash=corpus.manifest_hash(),
        scientific_hash=scientific.manifest_hash(),
        stopwords_hash=scientific.context_stopwords_hash(),
        index_definition_hash=index_definition_hash(scientific),
        document_count=1,
        chunk_count=1,
    )
    dense = _dense_manifest(scientific, corpus)

    wrong_revision = dense.model_copy(update={"model_revision": "0" * 40})
    low_recall = dense.model_copy(update={"recall_at_300": 0.97})

    assert verify_corpus_readiness(
        corpus, bm25, wrong_revision, scientific
    ).model_dump() == {
        "ready": False,
        "detail": "dense model revision mismatch",
    }
    assert verify_corpus_readiness(
        corpus, bm25, low_recall, scientific
    ).model_dump() == {
        "ready": False,
        "detail": "dense recall@300 is below 0.9800",
    }


def _dense_manifest(
    scientific: ScientificConfig,
    corpus: CorpusManifest,
) -> DenseIndexManifest:
    return DenseIndexManifest(
        model_name=scientific.embedding_model,
        model_revision=scientific.embedding_model_revision,
        embedding_dimension=scientific.embedding_dimension,
        corpus_manifest_hash=corpus.manifest_hash(),
        scientific_hash=scientific.manifest_hash(),
        chunks_sha256=corpus.chunks_sha256,
        embeddings_sha256="5" * 64,
        metadata_sha256="6" * 64,
        candidate_index_sha256="7" * 64,
        chunk_count=corpus.chunk_count,
        dtype="float16",
        metric="inner_product",
        candidate_index="faiss-hnsw-flat-v1",
        hnsw_m=scientific.dense_hnsw_m,
        ef_construction=scientific.dense_ef_construction,
        ef_search=scientific.dense_ef_search,
        candidate_count=scientific.dense_candidate_count,
        recall_at_300=0.99,
        recall_sample_sha256="8" * 64,
    )


def _compatible_best_available_settings(
    tmp_path: Path,
) -> tuple[
    Settings,
    BestAvailableCorpusManifest,
    Bm25IndexManifest,
    CompressedShardedDenseIndexManifest,
]:
    scientific = ScientificConfig()
    tmp_path.mkdir(parents=True, exist_ok=True)
    chunk = CorpusChunk(
        id=hashlib.sha256(b"best-available-chunk").hexdigest(),
        document_id="best-available-document",
        ordinal=0,
        text="public archive passage",
        word_count=3,
        title="Public archive title",
        corpus="historylab-unlabeled",
        source_url=None,
        classification=None,
        authored=None,
        doc_lang=None,
    )
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(chunk.model_dump_json() + "\n", encoding="utf-8")
    chunks_sha256 = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    corpus = BestAvailableCorpusManifest(
        source_profile_hash="1" * 64,
        source_sha256="2" * 64,
        source_size_bytes=100,
        audit_manifest_hash="3" * 64,
        limitations_sha256="4" * 64,
        scientific_hash=scientific.manifest_hash(),
        chunker_version=scientific.corpus_chunker_version,
        chunk_words=scientific.chunk_words,
        chunk_overlap=scientific.chunk_overlap,
        parsed_row_count=1,
        included_row_count=1,
        empty_body_count=0,
        malformed_row_count=0,
        document_count=1,
        chunk_count=1,
        corpus_counts={"historylab-unlabeled": 1},
        chunks_sha256=chunks_sha256,
    )
    bm25 = Bm25IndexManifest(
        index_name=BEST_AVAILABLE_BM25_INDEX_NAME,
        retrieval_profile="best_available_foiarchive_v1",
        opensearch_version=scientific.opensearch_version,
        corpus_manifest_hash=corpus.manifest_hash(),
        scientific_hash=scientific.manifest_hash(),
        stopwords_hash=scientific.context_stopwords_hash(),
        index_definition_hash=index_definition_hash(scientific),
        document_count=1,
        chunk_count=1,
    )
    dense_root = tmp_path / "best-dense"
    shard_directory = dense_root / "shard-00000"
    shard_directory.mkdir(parents=True)
    profile = DenseArtifactProfile()
    offsets = dense_root / "chunks.offsets.u64"
    offsets.write_bytes(b"\0" * 8)
    template = dense_root / "candidate-template.faiss"
    _write_faiss_header_fixture(template, profile=profile, kind="template", ntotal=0)
    recall_report = dense_root / "recall-report.json"
    embeddings = shard_directory / "embeddings.i8"
    embeddings.write_bytes(b"\1" * scientific.embedding_dimension)
    candidate = shard_directory / "candidate.faiss"
    _write_faiss_header_fixture(candidate, profile=profile, kind="candidate", ntotal=1)
    template_sha256 = hashlib.sha256(template.read_bytes()).hexdigest()
    shard = CompressedDenseShardManifest(
        artifact_profile_hash=profile.manifest_hash(),
        candidate_template_sha256=template_sha256,
        shard_id=0,
        path=shard_directory.name,
        start_row=0,
        end_row=1,
        chunk_count=1,
        embeddings_size_bytes=embeddings.stat().st_size,
        embeddings_sha256=hashlib.sha256(embeddings.read_bytes()).hexdigest(),
        candidate_index_size_bytes=candidate.stat().st_size,
        candidate_index_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
    )
    queries = "".join(f"public certification query {index}\n" for index in range(100))
    queries_path = tmp_path / "best-recall-queries.txt"
    queries_path.write_text(queries, encoding="utf-8")
    query_sha256 = hashlib.sha256(queries.encode("utf-8")).hexdigest()
    recall_report.write_text(
        DenseRecallReport(
            artifact_profile_hash=profile.manifest_hash(),
            corpus_manifest_hash=corpus.manifest_hash(),
            recall_queries_sha256=query_sha256,
            recall_query_count=100,
            recall_at_300=0.99,
            per_query_recall=[0.99] * 100,
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    total_bytes = (
        offsets.stat().st_size
        + template.stat().st_size
        + recall_report.stat().st_size
        + embeddings.stat().st_size
        + candidate.stat().st_size
    )
    dense = CompressedShardedDenseIndexManifest(
        model_name=scientific.embedding_model,
        model_revision=scientific.embedding_model_revision,
        embedding_dimension=scientific.embedding_dimension,
        corpus_manifest_hash=corpus.manifest_hash(),
        dense_semantics_hash=dense_semantics_hash(scientific),
        chunks_sha256=corpus.chunks_sha256,
        chunk_count=1,
        artifact_profile=profile,
        artifact_profile_hash=profile.manifest_hash(),
        chunk_offsets_path=offsets.name,
        chunk_offsets_format=profile.metadata_lookup_version,
        chunk_offsets_size_bytes=offsets.stat().st_size,
        chunk_offsets_sha256=hashlib.sha256(offsets.read_bytes()).hexdigest(),
        candidate_template_path=template.name,
        candidate_template_size_bytes=template.stat().st_size,
        candidate_template_sha256=template_sha256,
        shard_size_limit=1_000_000,
        shards=[shard],
        embedding_file="embeddings.i8",
        dtype="int8",
        metric="inner_product",
        candidate_index=profile.candidate_index,
        recall_report_path=recall_report.name,
        recall_report_size_bytes=recall_report.stat().st_size,
        recall_report_sha256=hashlib.sha256(
            recall_report.read_bytes()
        ).hexdigest(),
        recall_at_300=0.99,
        recall_queries_sha256=query_sha256,
        recall_query_count=100,
        total_artifact_bytes=total_bytes,
    )
    (shard_directory / "manifest.json").write_text(
        shard.model_dump_json() + "\n",
        encoding="utf-8",
    )
    latency = DenseLatencyReport(
        dense_manifest_hash=dense.manifest_hash(),
        faiss_verification=DenseFaissVerification(
            dense_manifest_hash=dense.manifest_hash(),
            faiss_version=dense.artifact_profile.faiss_version,
            template_ntotal=0,
            candidate_shard_count=len(dense.shards),
            candidate_row_count=dense.chunk_count,
        ),
        recall_queries_sha256=dense.recall_queries_sha256,
        query_count=100,
        warm_search_seconds=[1.0] * 100,
        max_observed_seconds=1.0,
        maximum_allowed_seconds=profile.max_warm_search_seconds,
        platform=runtime_platform.platform(),
        machine=runtime_platform.machine(),
        runtime_environment="test-worker",
        worker_build_id="a" * 64,
        serving_host_id="b" * 64,
    )
    corpus_path = tmp_path / "best-corpus.json"
    bm25_path = tmp_path / "best-bm25.json"
    dense_path = tmp_path / "best-dense.json"
    latency_path = tmp_path / "best-latency.json"
    corpus_path.write_text(corpus.model_dump_json(), encoding="utf-8")
    bm25_path.write_text(bm25.model_dump_json(), encoding="utf-8")
    dense_path.write_text(dense.model_dump_json(), encoding="utf-8")
    latency_path.write_text(latency.model_dump_json(), encoding="utf-8")
    settings = Settings(
        best_available_corpus_manifest_path=corpus_path,
        best_available_bm25_manifest_path=bm25_path,
        best_available_dense_manifest_path=dense_path,
        best_available_dense_artifact_directory=dense_root,
        best_available_chunks_path=chunks_path,
        best_available_recall_queries_path=queries_path,
        best_available_latency_report_path=latency_path,
        opensearch_url="http://search.test:9200",
    )
    return settings, corpus, bm25, dense


def _best_available_test_contract(
    corpus: BestAvailableCorpusManifest,
    dense: CompressedShardedDenseIndexManifest,
) -> BestAvailableRetrievalContract:
    return BestAvailableRetrievalContract(
        corpus_manifest_hash=corpus.manifest_hash(),
        chunks_sha256=corpus.chunks_sha256,
        parsed_row_count=corpus.parsed_row_count,
        document_count=corpus.document_count,
        chunk_count=corpus.chunk_count,
        application_queries_sha256="0" * 64,
        recall_queries_sha256=dense.recall_queries_sha256,
        recall_query_count=dense.recall_query_count,
        latency_runtime_environment="test-worker",
    )


def _write_faiss_header_fixture(
    path: Path,
    *,
    profile: DenseArtifactProfile,
    kind: str,
    ntotal: int,
) -> None:
    payload = f"{kind}-payload".encode("utf-8")
    metadata = {
        "schema_version": "faiss-ivfpq-artifact-v1",
        "kind": kind,
        "faiss_version": profile.faiss_version,
        "artifact_profile_hash": profile.manifest_hash(),
        "faiss_layout_hash": faiss_layout_hash(profile),
        "dimension": profile.embedding_dimension,
        "ivf_nlist": profile.ivf_nlist,
        "pq_subvectors": profile.pq_subvectors,
        "pq_bits": profile.pq_bits,
        "metric": profile.metric,
        "is_trained": True,
        "ntotal": ntotal,
        "payload_size_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(b"RDIVFPQ1\n" + struct.pack("<I", len(header)) + header + payload)


def _rewrite_dense_fixture(
    settings: Settings,
    dense: CompressedShardedDenseIndexManifest,
) -> None:
    dense_path = settings.best_available_dense_manifest_path
    latency_path = settings.best_available_latency_report_path
    assert dense_path is not None
    assert latency_path is not None
    dense_path.write_text(dense.model_dump_json(), encoding="utf-8")
    latency_path.write_text(
        DenseLatencyReport(
            dense_manifest_hash=dense.manifest_hash(),
            faiss_verification=DenseFaissVerification(
                dense_manifest_hash=dense.manifest_hash(),
                faiss_version=dense.artifact_profile.faiss_version,
                template_ntotal=0,
                candidate_shard_count=len(dense.shards),
                candidate_row_count=dense.chunk_count,
            ),
            recall_queries_sha256=dense.recall_queries_sha256,
            query_count=dense.recall_query_count,
            warm_search_seconds=[1.0] * dense.recall_query_count,
            max_observed_seconds=1.0,
            maximum_allowed_seconds=dense.artifact_profile.max_warm_search_seconds,
            platform=runtime_platform.platform(),
            machine=runtime_platform.machine(),
            runtime_environment="test-worker",
            worker_build_id="a" * 64,
            serving_host_id="b" * 64,
        ).model_dump_json(),
        encoding="utf-8",
    )
