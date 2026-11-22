from __future__ import annotations

import hashlib
import json
import builtins
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.retrieval.dense import DenseRecallError, require_dense_recall
from redacted_backend.retrieval.dense_profile import (
    DenseArtifactProfile,
    dense_semantics_hash,
)
from redacted_backend.retrieval.dense_sharded import (
    _best_scored_rows,
    _matrix_scores,
    certify_sharded_dense_recall,
    select_certification_queries,
)
from redacted_backend.corpus.build_dense_sharded import _best_scores
from redacted_backend.retrieval.types import (
    CompressedDenseShardManifest,
    CompressedShardedDenseIndexManifest,
)
from redacted_backend.scientific_config import ScientificConfig


class Encoder:
    def encode_query(self, text: str) -> NDArray[np.float32]:
        del text
        return np.asarray([1.0, 0.0], dtype=np.float32)


class Candidate:
    def __init__(self, rows: list[int], *, ntotal: int) -> None:
        self.rows = rows
        self.ntotal = ntotal

    def search(
        self,
        query: NDArray[np.float32],
        candidates: int,
    ) -> NDArray[np.int64]:
        del query
        return np.asarray(self.rows[:candidates], dtype=np.int64)


def test_builder_certifier_and_runtime_use_global_row_tie_breaking() -> None:
    tied = [(0.75, 9), (0.75, 2), (0.75, 5)]

    assert _best_scores(tied, 2) == [(0.75, 2), (0.75, 5)]
    assert _best_scored_rows(tied, 2) == [(0.75, 2), (0.75, 5)]


def _chunk(index: int) -> CorpusChunk:
    text = f"opening archive words {index}"
    return CorpusChunk(
        id=hashlib.sha256(f"chunk-{index}".encode()).hexdigest(),
        document_id=f"document-{index}",
        ordinal=0,
        text=text,
        word_count=4,
        title=f"Public title {index}",
        corpus="historylab-unlabeled",
        source_url=None,
        classification=None,
        authored=None,
        doc_lang=None,
    )


def test_certification_query_selection_is_stable_and_hashes_published_bytes(
    tmp_path: Path,
) -> None:
    chunks = [_chunk(index) for index in range(5)]
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        "".join(chunk.model_dump_json() + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    application = tmp_path / "application.txt"
    application.write_text(
        "visible public query A\nvisible public query B\n",
        encoding="utf-8",
    )
    output = tmp_path / "combined.txt"

    queries = select_certification_queries(
        chunks_path,
        application,
        output,
        application_query_count=2,
        corpus_query_count=2,
    )

    assert queries == [
        "visible public query A",
        "visible public query B",
        "Public title 1 opening archive words 1",
        "Public title 0 opening archive words 0",
    ]
    assert output.read_text(encoding="utf-8") == "\n".join(queries) + "\n"
    assert hashlib.sha256(output.read_bytes()).hexdigest() == (
        "7eb395ce98f7149e54fdcd20e9a173fd168418b3489442815cd214729ce99c41"
    )


def _write_recall_fixture(
    root: Path,
) -> tuple[CompressedShardedDenseIndexManifest, dict[str, list[int]]]:
    scientific = ScientificConfig(embedding_dimension=2)
    profile = DenseArtifactProfile(
        embedding_dimension=2,
        ivf_nlist=1,
        pq_subvectors=1,
        pq_bits=1,
        nprobe=1,
        candidate_count_per_shard=2,
        training_sample_count=4,
        shard_size_limit=2,
        max_artifact_bytes=1024 * 1024,
    )
    template = root / "candidate-template.faiss"
    template.write_bytes(b"template")
    template_sha256 = hashlib.sha256(template.read_bytes()).hexdigest()
    offsets = root / "chunks.offsets.u64"
    offsets.write_bytes(np.asarray([0, 1, 2, 3], dtype="<u8").tobytes())
    recall = root / "recall-report.json"
    recall.write_bytes(b"{}\n")
    rows_by_path: dict[str, list[int]] = {}
    shards: list[CompressedDenseShardManifest] = []
    for shard_id, (scores, rows) in enumerate(
        (([0.9, 0.1], [0]), ([0.8, 0.7], [0, 1]))
    ):
        directory = root / f"shard-{shard_id:05d}"
        directory.mkdir()
        embeddings = directory / "embeddings.f16"
        np.asarray(
            [[score, 1.0 - score] for score in scores],
            dtype=np.float16,
        ).tofile(embeddings)
        candidate = directory / "candidate.faiss"
        candidate.write_text(json.dumps(rows), encoding="utf-8")
        rows_by_path[directory.name] = rows
        shards.append(
            CompressedDenseShardManifest(
                artifact_profile_hash=profile.manifest_hash(),
                candidate_template_sha256=template_sha256,
                shard_id=shard_id,
                path=directory.name,
                start_row=shard_id * 2,
                end_row=shard_id * 2 + 2,
                chunk_count=2,
                embeddings_size_bytes=embeddings.stat().st_size,
                embeddings_sha256=hashlib.sha256(
                    embeddings.read_bytes()
                ).hexdigest(),
                candidate_index_size_bytes=candidate.stat().st_size,
                candidate_index_sha256=hashlib.sha256(
                    candidate.read_bytes()
                ).hexdigest(),
            )
        )
    listed_bytes = (
        offsets.stat().st_size
        + template.stat().st_size
        + recall.stat().st_size
        + sum(
            shard.embeddings_size_bytes + shard.candidate_index_size_bytes
            for shard in shards
        )
    )
    return (
        CompressedShardedDenseIndexManifest(
            model_name=scientific.embedding_model,
            model_revision=scientific.embedding_model_revision,
            embedding_dimension=2,
            corpus_manifest_hash="1" * 64,
            dense_semantics_hash=dense_semantics_hash(scientific),
            chunks_sha256="3" * 64,
            chunk_count=4,
            artifact_profile=profile,
            artifact_profile_hash=profile.manifest_hash(),
            chunk_offsets_path=offsets.name,
            chunk_offsets_format=profile.metadata_lookup_version,
            chunk_offsets_size_bytes=offsets.stat().st_size,
            chunk_offsets_sha256=hashlib.sha256(offsets.read_bytes()).hexdigest(),
            candidate_template_path=template.name,
            candidate_template_size_bytes=template.stat().st_size,
            candidate_template_sha256=template_sha256,
            shard_size_limit=2,
            shards=shards,
            dtype="float16",
            metric="inner_product",
            candidate_index=profile.candidate_index,
            recall_report_path=recall.name,
            recall_report_size_bytes=recall.stat().st_size,
            recall_report_sha256=hashlib.sha256(recall.read_bytes()).hexdigest(),
            recall_at_300=1.0,
            recall_queries_sha256="4" * 64,
            recall_query_count=1,
            total_artifact_bytes=listed_bytes,
        ),
        rows_by_path,
    )


def test_sharded_recall_compares_candidate_union_with_global_exact_top_k(
    tmp_path: Path,
) -> None:
    manifest, rows_by_path = _write_recall_fixture(tmp_path)

    def loader(path: Path, *, profile: DenseArtifactProfile) -> Candidate:
        assert profile == manifest.artifact_profile
        return Candidate(
            rows_by_path[path.parent.name],
            ntotal=manifest.shards[int(path.parent.name[-5:])].chunk_count,
        )

    recall = certify_sharded_dense_recall(
        tmp_path,
        manifest,
        ["query"],
        encoder=Encoder(),
        candidate_loader=loader,
        scientific=ScientificConfig(embedding_dimension=2),
        limit=2,
        block_rows=1,
        matrix_backend="numpy",
    )

    assert recall == 1.0


def test_auto_matrix_backend_avoids_torch_without_visible_cuda_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    queries = np.asarray([[1.0, 0.0]], dtype=np.float32)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("auto mode imported torch without a CUDA runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(
        "redacted_backend.retrieval.dense_sharded._cuda_runtime_visible",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    scores = _matrix_scores(embeddings, queries, "auto")

    assert scores.tolist() == [[1.0], [0.0]]


def test_recall_gate_publishes_at_point_98_and_refuses_below_it() -> None:
    require_dense_recall(0.98, minimum=0.98)
    with pytest.raises(DenseRecallError, match="0.9799"):
        require_dense_recall(0.9799, minimum=0.98)
