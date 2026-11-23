from __future__ import annotations

import argparse
import heapq
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from redacted_backend.retrieval.dense_profile import DenseArtifactProfile
from redacted_backend.retrieval.faiss_ivfpq import (
    FaissIvfPqIndex,
    train_ivfpq_template,
)
from redacted_backend.retrieval.int8_embeddings import dequantize_directions


class FaissBenchmarkJob(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["faiss-benchmark-job-v1"] = "faiss-benchmark-job-v1"
    embeddings_path: Path
    output_directory: Path
    sample_rows: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    top_chunks_per_list: int = Field(ge=1)
    profile: DenseArtifactProfile


class FaissBenchmarkResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["faiss-benchmark-result-v1"] = (
        "faiss-benchmark-result-v1"
    )
    torch_imported: Literal[False] = False
    training_seconds: float = Field(ge=0)
    candidate_add_seconds: float = Field(ge=0)
    certification_probe_seconds: float = Field(ge=0)
    certification_probe_queries: int = Field(ge=1)


class FaissProductionJob(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["faiss-production-job-v1"] = "faiss-production-job-v1"
    output_directory: Path
    checkpoints: list[dict[str, object]] = Field(min_length=1)
    training_rows: dict[str, object]
    profile: DenseArtifactProfile
    query_vectors_path: Path
    query_count: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    recall_block_rows: int = Field(ge=1)
    top_chunks_per_list: int = Field(ge=1)


class FaissProductionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["faiss-production-result-v1"] = (
        "faiss-production-result-v1"
    )
    torch_imported: Literal[False] = False
    template: dict[str, object]
    shards: list[dict[str, object]] = Field(min_length=1)
    per_query_recall: list[float] = Field(min_length=1)


def run_faiss_benchmark_job(job: FaissBenchmarkJob) -> FaissBenchmarkResult:
    """Execute all FAISS work in a fresh interpreter that never imports Torch."""

    job.output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=job.output_directory,
        prefix=".faiss-worker-",
    ) as temporary_name:
        temporary = Path(temporary_name)
        job_path = temporary / "job.json"
        result_path = temporary / "result.json"
        _write_json(job_path, job.model_dump(mode="json"))
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "redacted_backend.corpus.faiss_build_worker",
                "--mode",
                "benchmark",
                "--job",
                str(job_path),
                "--result",
                str(result_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                "Isolated FAISS benchmark failed "
                f"with exit code {completed.returncode}: {detail}"
            )
        if not result_path.is_file():
            raise RuntimeError("Isolated FAISS benchmark did not publish a result")
        return FaissBenchmarkResult.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )


def run_faiss_production_job(job: FaissProductionJob) -> FaissProductionResult:
    """Build and certify production FAISS artifacts outside the Torch process."""

    job.output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=job.output_directory,
        prefix=".faiss-worker-",
    ) as temporary_name:
        temporary = Path(temporary_name)
        job_path = temporary / "job.json"
        result_path = temporary / "result.json"
        _write_json(job_path, job.model_dump(mode="json"))
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "redacted_backend.corpus.faiss_build_worker",
                "--mode",
                "production",
                "--job",
                str(job_path),
                "--result",
                str(result_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                "Isolated production FAISS build failed "
                f"with exit code {completed.returncode}: {detail}"
            )
        if not result_path.is_file():
            raise RuntimeError("Isolated production FAISS build did not publish a result")
        return FaissProductionResult.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )


def _execute_faiss_benchmark_job(job: FaissBenchmarkJob) -> FaissBenchmarkResult:
    if "torch" in sys.modules:
        raise RuntimeError("FAISS build worker must not import Torch")
    expected_size = job.sample_rows * job.profile.embedding_dimension
    if (
        not job.embeddings_path.is_file()
        or job.embeddings_path.stat().st_size != expected_size
    ):
        raise ValueError("FAISS benchmark embedding size does not match the job")
    job.output_directory.mkdir(parents=True, exist_ok=True)
    embeddings = np.memmap(
        job.embeddings_path,
        dtype=np.int8,
        mode="r",
        shape=(job.sample_rows, job.profile.embedding_dimension),
    )

    training_started = time.monotonic()
    training_vectors = dequantize_directions(np.asarray(embeddings, dtype=np.int8))
    template_path = job.output_directory / "candidate-template.faiss"
    train_ivfpq_template(
        training_vectors,
        template_path,
        dimension=job.profile.embedding_dimension,
        profile=job.profile,
    )
    training_seconds = time.monotonic() - training_started

    candidate = FaissIvfPqIndex.create(template_path, profile=job.profile)
    candidate.save(job.output_directory / "empty-candidate.faiss")
    candidate_add_started = time.monotonic()
    for start in range(0, job.sample_rows, job.batch_size):
        end = min(start + job.batch_size, job.sample_rows)
        candidate.add(training_vectors[start:end])
    candidate.save(job.output_directory / "candidate.faiss")
    candidate_add_seconds = time.monotonic() - candidate_add_started

    probe_query_count = min(10, job.sample_rows)
    certification_started = time.monotonic()
    probe_queries = np.ascontiguousarray(
        training_vectors[:probe_query_count],
        dtype=np.float32,
    )
    top_k = min(job.top_chunks_per_list, job.sample_rows)
    exact_probe: list[list[tuple[float, int]]] = [
        [] for _ in range(probe_query_count)
    ]
    for start in range(0, job.sample_rows, 16_384):
        end = min(start + 16_384, job.sample_rows)
        scores = training_vectors[start:end] @ probe_queries.T
        for query_index in range(probe_query_count):
            values = [
                (float(score), start + row)
                for row, score in enumerate(scores[:, query_index])
            ]
            exact_probe[query_index] = heapq.nlargest(
                top_k,
                (*exact_probe[query_index], *values),
                key=lambda value: (value[0], -value[1]),
            )
    for query_vector in probe_queries:
        candidate.search(
            query_vector,
            min(job.profile.candidate_count_per_shard, job.sample_rows),
        )
    certification_probe_seconds = time.monotonic() - certification_started
    del embeddings
    if "torch" in sys.modules:
        raise RuntimeError("FAISS build worker imported Torch unexpectedly")
    return FaissBenchmarkResult(
        training_seconds=training_seconds,
        candidate_add_seconds=candidate_add_seconds,
        certification_probe_seconds=certification_probe_seconds,
        certification_probe_queries=probe_query_count,
    )


def _execute_faiss_production_job(job: FaissProductionJob) -> FaissProductionResult:
    if "torch" in sys.modules:
        raise RuntimeError("FAISS build worker must not import Torch")
    from redacted_backend.corpus.build_dense_sharded import (
        EmbeddingShardCheckpoint,
        TrainingRowsManifest,
        _build_candidate_shards,
        _certify_candidate_recall_vectors,
        _train_candidate_template,
    )

    checkpoints = [
        EmbeddingShardCheckpoint.model_validate(value) for value in job.checkpoints
    ]
    training_rows = TrainingRowsManifest.model_validate(job.training_rows)
    expected_query_size = (
        job.query_count * job.profile.embedding_dimension * np.dtype(np.float32).itemsize
    )
    if (
        not job.query_vectors_path.is_file()
        or job.query_vectors_path.stat().st_size != expected_query_size
    ):
        raise ValueError("FAISS production query-vector size does not match the job")
    query_vectors = np.memmap(
        job.query_vectors_path,
        dtype=np.float32,
        mode="r",
        shape=(job.query_count, job.profile.embedding_dimension),
    )
    template = _train_candidate_template(
        job.output_directory,
        checkpoints=checkpoints,
        training_rows=training_rows,
        profile=job.profile,
    )
    shards = _build_candidate_shards(
        job.output_directory,
        checkpoints=checkpoints,
        template=template,
        profile=job.profile,
        batch_size=job.batch_size,
    )
    per_query_recall = _certify_candidate_recall_vectors(
        job.output_directory,
        shards,
        np.ascontiguousarray(query_vectors, dtype=np.float32),
        profile=job.profile,
        block_rows=job.recall_block_rows,
        top_chunks_per_list=job.top_chunks_per_list,
    )
    del query_vectors
    if "torch" in sys.modules:
        raise RuntimeError("FAISS build worker imported Torch unexpectedly")
    return FaissProductionResult(
        template=template.model_dump(mode="json"),
        shards=[item.model_dump(mode="json") for item in shards],
        per_query_recall=per_query_recall,
    )


def _write_json(path: Path, value: object) -> None:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("benchmark", "production"), required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    payload = Path(args.job).read_text(encoding="utf-8")
    if args.mode == "benchmark":
        result = _execute_faiss_benchmark_job(
            FaissBenchmarkJob.model_validate_json(payload)
        )
    else:
        result = _execute_faiss_production_job(
            FaissProductionJob.model_validate_json(payload)
        )
    _write_json(Path(args.result), result.model_dump(mode="json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
