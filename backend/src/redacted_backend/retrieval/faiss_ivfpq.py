from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Literal, Self

import faiss
import numpy as np
from numpy.typing import NDArray

from redacted_backend.retrieval.dense import IncompatibleDenseIndexError
from redacted_backend.retrieval.dense_profile import DenseArtifactProfile


_ARTIFACT_MAGIC = b"RDIVFPQ1\n"
_MAX_HEADER_BYTES = 1024 * 1024
_ArtifactKind = Literal["template", "candidate"]


def train_ivfpq_template(
    training_vectors: NDArray[np.float32],
    output_path: Path,
    *,
    dimension: int,
    profile: DenseArtifactProfile,
) -> str:
    if dimension != profile.embedding_dimension:
        raise ValueError("Training dimension does not match the artifact profile")
    vectors = _validated_matrix(
        training_vectors,
        dimension=dimension,
        label="Training vectors",
    )
    if len(vectors) != profile.training_sample_count:
        raise ValueError(
            "Training vector count must equal the artifact profile sample count"
        )
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFPQ(
        quantizer,
        dimension,
        profile.ivf_nlist,
        profile.pq_subvectors,
        profile.pq_bits,
        faiss.METRIC_INNER_PRODUCT,
    )
    index.cp.seed = profile.training_seed
    index.pq.cp.seed = profile.training_seed
    index.train(vectors)
    if not index.is_trained:
        raise IncompatibleDenseIndexError("FAISS IVF-PQ template did not become trained")
    _write_faiss_artifact(
        output_path,
        index=index,
        kind="template",
        profile=profile,
    )
    return _sha256_file(output_path)


class FaissIvfPqIndex:
    def __init__(self, index: object, profile: DenseArtifactProfile) -> None:
        self._index = index
        self._profile = profile

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    @property
    def nprobe(self) -> int:
        return int(self._index.nprobe)

    @classmethod
    def create(
        cls,
        template_path: Path,
        *,
        profile: DenseArtifactProfile,
    ) -> Self:
        template = _read_faiss_artifact(
            template_path,
            expected_kind="template",
            profile=profile,
        )
        if not template.is_trained:
            raise IncompatibleDenseIndexError("FAISS IVF-PQ template is not trained")
        if template.ntotal != 0:
            raise IncompatibleDenseIndexError(
                "FAISS IVF-PQ template must not contain candidate rows"
            )
        index = faiss.clone_index(template)
        index.reset()
        index.nprobe = profile.nprobe
        return cls(index, profile)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        profile: DenseArtifactProfile,
    ) -> Self:
        index = _read_faiss_artifact(
            path,
            expected_kind="candidate",
            profile=profile,
        )
        if not index.is_trained:
            raise IncompatibleDenseIndexError("FAISS IVF-PQ candidate is not trained")
        if index.ntotal > profile.shard_size_limit:
            raise IncompatibleDenseIndexError(
                "FAISS IVF-PQ candidate exceeds the shard limit"
            )
        index.nprobe = profile.nprobe
        return cls(index, profile)

    def add(self, vectors: NDArray[np.float32]) -> None:
        matrix = _validated_matrix(
            vectors,
            dimension=self._profile.embedding_dimension,
            label="Candidate vectors",
        )
        if self.ntotal + len(matrix) > self._profile.shard_size_limit:
            raise ValueError("Candidate rows exceed the artifact profile shard limit")
        self._index.add(matrix)

    def search(
        self,
        query: NDArray[np.float32],
        candidates: int,
    ) -> NDArray[np.int64]:
        if candidates < 1 or candidates > self._profile.candidate_count_per_shard:
            raise ValueError("Requested candidate count is outside the artifact profile")
        vector = _validated_query(
            query,
            dimension=self._profile.embedding_dimension,
        )
        _scores, raw_rows = self._index.search(vector.reshape(1, -1), candidates)
        rows: list[int] = []
        seen: set[int] = set()
        for value in np.asarray(raw_rows, dtype=np.int64).reshape(-1):
            row = int(value)
            if row == -1 or row in seen:
                continue
            if row < 0 or row >= self.ntotal:
                raise IncompatibleDenseIndexError(
                    "FAISS IVF-PQ returned an invalid local row"
                )
            seen.add(row)
            rows.append(row)
        return np.asarray(rows, dtype=np.int64)

    def save(self, path: Path) -> None:
        _write_faiss_artifact(
            path,
            index=self._index,
            kind="candidate",
            profile=self._profile,
        )


def _validated_matrix(
    vectors: NDArray[np.float32],
    *,
    dimension: int,
    label: str,
) -> NDArray[np.float32]:
    matrix = np.asarray(vectors)
    if matrix.dtype != np.float32:
        raise ValueError(f"{label} must use float32 values")
    if matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise ValueError(f"{label} dimension does not match the artifact profile")
    if matrix.shape[0] < 1:
        raise ValueError(f"{label} must not be empty")
    if np.any(~np.isfinite(matrix)):
        raise ValueError(f"{label} must be finite")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
        raise ValueError(f"{label} must be normalized")
    return np.ascontiguousarray(matrix)


def _validated_query(
    query: NDArray[np.float32],
    *,
    dimension: int,
) -> NDArray[np.float32]:
    vector = np.asarray(query)
    if vector.dtype != np.float32:
        raise ValueError("Dense query must use float32 values")
    if vector.shape != (dimension,):
        raise ValueError("Dense query dimension does not match the artifact profile")
    if np.any(~np.isfinite(vector)):
        raise ValueError("Dense query must be finite")
    if not np.isclose(np.linalg.norm(vector), 1.0, rtol=1e-4, atol=1e-4):
        raise ValueError("Dense query must be normalized")
    return np.ascontiguousarray(vector)


def _write_faiss_artifact(
    path: Path,
    *,
    index: object,
    kind: _ArtifactKind,
    profile: DenseArtifactProfile,
    faiss_version: str | None = None,
) -> None:
    serialized = np.asarray(faiss.serialize_index(index), dtype=np.uint8)
    payload = serialized.tobytes()
    metadata = {
        "schema_version": "faiss-ivfpq-artifact-v1",
        "kind": kind,
        "faiss_version": faiss_version or faiss.__version__,
        "artifact_profile_hash": profile.manifest_hash(),
        "dimension": int(index.d),
        "ivf_nlist": int(index.nlist),
        "pq_subvectors": int(index.pq.M),
        "pq_bits": int(index.pq.nbits),
        "metric": "inner_product"
        if index.metric_type == faiss.METRIC_INNER_PRODUCT
        else str(index.metric_type),
        "is_trained": bool(index.is_trained),
        "ntotal": int(index.ntotal),
        "payload_size_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    header = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(header) > _MAX_HEADER_BYTES:
        raise ValueError("FAISS artifact metadata is unexpectedly large")
    value = _ARTIFACT_MAGIC + struct.pack("<I", len(header)) + header + payload
    _write_bytes_atomic(path, value)


def _read_faiss_artifact(
    path: Path,
    *,
    expected_kind: _ArtifactKind,
    profile: DenseArtifactProfile,
) -> object:
    try:
        value = path.read_bytes()
    except OSError as error:
        raise IncompatibleDenseIndexError(
            f"Cannot read FAISS IVF-PQ artifact: {path}"
        ) from error
    prefix_size = len(_ARTIFACT_MAGIC) + 4
    if len(value) < prefix_size or not value.startswith(_ARTIFACT_MAGIC):
        raise IncompatibleDenseIndexError("FAISS IVF-PQ artifact format is invalid")
    header_size = struct.unpack(
        "<I",
        value[len(_ARTIFACT_MAGIC) : prefix_size],
    )[0]
    if header_size < 2 or header_size > _MAX_HEADER_BYTES:
        raise IncompatibleDenseIndexError("FAISS IVF-PQ metadata size is invalid")
    payload_start = prefix_size + header_size
    if payload_start > len(value):
        raise IncompatibleDenseIndexError("FAISS IVF-PQ artifact is truncated")
    try:
        metadata = json.loads(value[prefix_size:payload_start])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IncompatibleDenseIndexError("FAISS IVF-PQ metadata is invalid") from error
    if not isinstance(metadata, dict):
        raise IncompatibleDenseIndexError("FAISS IVF-PQ metadata must be an object")
    payload = value[payload_start:]
    _validate_metadata(metadata, payload, expected_kind=expected_kind, profile=profile)
    try:
        index = faiss.deserialize_index(np.frombuffer(payload, dtype=np.uint8).copy())
    except RuntimeError as error:
        raise IncompatibleDenseIndexError("FAISS IVF-PQ payload is invalid") from error
    _validate_index(index, metadata, profile)
    index.nprobe = profile.nprobe
    return index


def _validate_metadata(
    metadata: dict[str, object],
    payload: bytes,
    *,
    expected_kind: _ArtifactKind,
    profile: DenseArtifactProfile,
) -> None:
    if metadata.get("schema_version") != "faiss-ivfpq-artifact-v1":
        raise IncompatibleDenseIndexError("FAISS IVF-PQ artifact schema mismatch")
    if metadata.get("kind") != expected_kind:
        raise IncompatibleDenseIndexError("FAISS IVF-PQ artifact kind mismatch")
    if (
        metadata.get("faiss_version") != profile.faiss_version
        or faiss.__version__ != profile.faiss_version
    ):
        raise IncompatibleDenseIndexError("FAISS version does not match the profile")
    if metadata.get("artifact_profile_hash") != profile.manifest_hash():
        raise IncompatibleDenseIndexError("FAISS IVF-PQ artifact profile mismatch")
    if metadata.get("payload_size_bytes") != len(payload):
        raise IncompatibleDenseIndexError("FAISS IVF-PQ payload size mismatch")
    if metadata.get("payload_sha256") != hashlib.sha256(payload).hexdigest():
        raise IncompatibleDenseIndexError("FAISS IVF-PQ payload checksum mismatch")


def _validate_index(
    index: object,
    metadata: dict[str, object],
    profile: DenseArtifactProfile,
) -> None:
    if not isinstance(index, faiss.IndexIVFPQ):
        raise IncompatibleDenseIndexError("FAISS artifact is not an IVF-PQ index")
    expected = {
        "dimension": profile.embedding_dimension,
        "ivf_nlist": profile.ivf_nlist,
        "pq_subvectors": profile.pq_subvectors,
        "pq_bits": profile.pq_bits,
        "metric": profile.metric,
        "is_trained": bool(index.is_trained),
        "ntotal": int(index.ntotal),
    }
    actual = {
        "dimension": int(index.d),
        "ivf_nlist": int(index.nlist),
        "pq_subvectors": int(index.pq.M),
        "pq_bits": int(index.pq.nbits),
        "metric": "inner_product"
        if index.metric_type == faiss.METRIC_INNER_PRODUCT
        else str(index.metric_type),
        "is_trained": metadata.get("is_trained"),
        "ntotal": metadata.get("ntotal"),
    }
    if actual != expected:
        raise IncompatibleDenseIndexError("FAISS IVF-PQ index parameters mismatch")


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            descriptor = -1
            destination.write(value)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
