from __future__ import annotations

import hashlib
from pathlib import Path

import faiss
import numpy as np
import pytest
from numpy.typing import NDArray

from redacted_backend.retrieval.dense import IncompatibleDenseIndexError
from redacted_backend.retrieval.dense_profile import DenseArtifactProfile
from redacted_backend.retrieval.faiss_ivfpq import (
    FaissIvfPqIndex,
    _write_faiss_artifact,
    train_ivfpq_template,
)


def _profile(**updates: object) -> DenseArtifactProfile:
    values = {
        "embedding_dimension": 8,
        "ivf_nlist": 2,
        "pq_subvectors": 2,
        "pq_bits": 4,
        "nprobe": 2,
        "candidate_count_per_shard": 16,
        "training_sample_count": 1_024,
        "shard_size_limit": 1_024,
    }
    values.update(updates)
    return DenseArtifactProfile(**values)


def _unit_vectors(rows: int, dimension: int = 8) -> NDArray[np.float32]:
    values = np.random.default_rng(20261122).normal(size=(rows, dimension)).astype(
        np.float32
    )
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values


def test_trained_template_creates_independent_persistent_compressed_indexes(
    tmp_path: Path,
) -> None:
    profile = _profile()
    vectors = _unit_vectors(profile.training_sample_count)
    template_path = tmp_path / "template.faiss"

    assert profile.training_seed == 1_234
    template_sha256 = train_ivfpq_template(
        vectors,
        template_path,
        dimension=8,
        profile=profile,
    )
    first = FaissIvfPqIndex.create(template_path, profile=profile)
    second = FaissIvfPqIndex.create(template_path, profile=profile)

    assert template_sha256 == hashlib.sha256(template_path.read_bytes()).hexdigest()
    assert first.ntotal == second.ntotal == 0
    assert first.nprobe == second.nprobe == 2
    first.add(vectors)
    assert first.ntotal == 1_024
    assert second.ntotal == 0

    candidate_path = tmp_path / "candidate.faiss"
    first.save(candidate_path)
    loaded = FaissIvfPqIndex.load(candidate_path, profile=profile)
    before = first.search(vectors[17], candidates=16)
    after = loaded.search(vectors[17], candidates=16)

    assert np.array_equal(after, before)
    assert all(0 <= int(row) < len(vectors) for row in after)
    assert candidate_path.stat().st_size < vectors.nbytes


@pytest.mark.parametrize(
    "vectors",
    [
        np.ones((1_024, 7), dtype=np.float32),
        np.full((1_024, 8), np.nan, dtype=np.float32),
        np.full((1_024, 8), 0.5, dtype=np.float32),
    ],
)
def test_training_rejects_wrong_nonfinite_or_unnormalized_vectors(
    tmp_path: Path,
    vectors: NDArray[np.float32],
) -> None:
    with pytest.raises(ValueError, match="dimension|finite|normalized"):
        train_ivfpq_template(
            vectors,
            tmp_path / "template.faiss",
            dimension=8,
            profile=_profile(),
        )


def test_add_rejects_invalid_vectors_and_rows_beyond_the_shard_limit(
    tmp_path: Path,
) -> None:
    profile = _profile(shard_size_limit=4)
    training = _unit_vectors(profile.training_sample_count)
    template_path = tmp_path / "template.faiss"
    train_ivfpq_template(training, template_path, dimension=8, profile=profile)
    index = FaissIvfPqIndex.create(template_path, profile=profile)

    for invalid in (
        np.ones((1, 7), dtype=np.float32),
        np.full((1, 8), np.nan, dtype=np.float32),
        np.full((1, 8), 0.5, dtype=np.float32),
    ):
        with pytest.raises(ValueError, match="dimension|finite|normalized"):
            index.add(invalid)
    index.add(training[:3])
    with pytest.raises(ValueError, match="shard limit"):
        index.add(training[3:5])


def test_create_rejects_untrained_wrong_version_or_wrong_profile_templates(
    tmp_path: Path,
) -> None:
    profile = _profile()
    quantizer = faiss.IndexFlatIP(8)
    untrained = faiss.IndexIVFPQ(quantizer, 8, 2, 2, 4, faiss.METRIC_INNER_PRODUCT)
    untrained_path = tmp_path / "untrained.faiss"
    _write_faiss_artifact(
        untrained_path,
        index=untrained,
        kind="template",
        profile=profile,
    )
    with pytest.raises(IncompatibleDenseIndexError, match="trained"):
        FaissIvfPqIndex.create(untrained_path, profile=profile)

    wrong_version_path = tmp_path / "wrong-version.faiss"
    _write_faiss_artifact(
        wrong_version_path,
        index=untrained,
        kind="template",
        profile=profile,
        faiss_version="0.0.0",
    )
    with pytest.raises(IncompatibleDenseIndexError, match="FAISS version"):
        FaissIvfPqIndex.create(wrong_version_path, profile=profile)

    trained_path = tmp_path / "trained.faiss"
    train_ivfpq_template(
        _unit_vectors(profile.training_sample_count),
        trained_path,
        dimension=8,
        profile=profile,
    )
    with pytest.raises(IncompatibleDenseIndexError, match="profile"):
        FaissIvfPqIndex.create(
            trained_path,
            profile=profile.model_copy(update={"nprobe": 1}),
        )


class _DuplicateRowsIndex:
    ntotal = 3
    nprobe = 2

    def search(
        self,
        query: NDArray[np.float32],
        candidates: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        del query, candidates
        return (
            np.asarray([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32),
            np.asarray([[2, -1, 2, 0]], dtype=np.int64),
        )


def test_search_filters_sentinels_and_duplicates_in_faiss_rank_order() -> None:
    profile = _profile()
    index = object.__new__(FaissIvfPqIndex)
    index._index = _DuplicateRowsIndex()
    index._profile = profile

    assert index.search(_unit_vectors(1)[0], candidates=4).tolist() == [2, 0]


def test_search_rejects_invalid_queries_and_candidate_counts(tmp_path: Path) -> None:
    profile = _profile()
    training = _unit_vectors(profile.training_sample_count)
    template_path = tmp_path / "template.faiss"
    train_ivfpq_template(training, template_path, dimension=8, profile=profile)
    index = FaissIvfPqIndex.create(template_path, profile=profile)
    index.add(training[:4])

    for candidates in (0, profile.candidate_count_per_shard + 1):
        with pytest.raises(ValueError, match="candidate"):
            index.search(training[0], candidates=candidates)
    for query in (
        np.ones(7, dtype=np.float32),
        np.full(8, np.nan, dtype=np.float32),
        np.full(8, 0.5, dtype=np.float32),
    ):
        with pytest.raises(ValueError, match="dimension|finite|normalized"):
            index.search(query, candidates=4)
