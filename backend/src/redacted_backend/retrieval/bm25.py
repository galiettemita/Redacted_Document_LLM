from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import Any, Literal, Protocol
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.corpus.manifest import CorpusManifest
from redacted_backend.retrieval.types import Bm25IndexManifest, RankedChunk
from redacted_backend.scientific_config import ScientificConfig


class IncompatibleIndexManifestError(ValueError):
    pass


class SearchTransport(Protocol):
    def replace_index(self, index_name: str, definition: dict[str, Any]) -> None: ...

    def bulk_index(
        self,
        index_name: str,
        documents: Sequence[dict[str, Any]],
    ) -> None: ...

    def refresh(self, index_name: str) -> None: ...

    def search(
        self,
        index_name: str,
        request: dict[str, Any],
    ) -> list[dict[str, Any]]: ...


class Bm25Index:
    def __init__(
        self,
        transport: SearchTransport,
        *,
        scientific: ScientificConfig | None = None,
        index_manifest: Bm25IndexManifest | None = None,
        corpus_manifest: CorpusManifest | None = None,
    ) -> None:
        self.transport = transport
        self.scientific = scientific or ScientificConfig()
        self.index_manifest = index_manifest
        self.corpus_manifest = corpus_manifest

    def build(
        self,
        chunks: Iterable[CorpusChunk],
        manifest: CorpusManifest,
    ) -> Bm25IndexManifest:
        self._validate_corpus_manifest(manifest)
        definition = build_index_definition(self.scientific)
        self.transport.replace_index(self.scientific.bm25_index_name, definition)
        indexed_count = 0
        batch: list[dict[str, Any]] = []
        for chunk in chunks:
            batch.append(_index_document(chunk))
            indexed_count += 1
            if len(batch) >= self.scientific.bm25_bulk_batch_size:
                self.transport.bulk_index(self.scientific.bm25_index_name, batch)
                batch = []
        if batch:
            self.transport.bulk_index(self.scientific.bm25_index_name, batch)
        if indexed_count != manifest.chunk_count:
            raise IncompatibleIndexManifestError(
                "Indexed chunk count does not match the corpus manifest"
            )
        self.transport.refresh(self.scientific.bm25_index_name)

        index_manifest = Bm25IndexManifest(
            index_name=self.scientific.bm25_index_name,
            opensearch_version=self.scientific.opensearch_version,
            corpus_manifest_hash=manifest.manifest_hash(),
            scientific_hash=self.scientific.manifest_hash(),
            index_definition_hash=index_definition_hash(self.scientific),
            stopwords_hash=self.scientific.context_stopwords_hash(),
            document_count=manifest.document_count,
            chunk_count=indexed_count,
        )
        self.index_manifest = index_manifest
        self.corpus_manifest = manifest
        return index_manifest

    def search(
        self,
        query: str,
        limit: int = 300,
        *,
        query_kind: Literal["context", "own"] = "context",
    ) -> list[RankedChunk]:
        if limit < 1:
            raise ValueError("Search limit must be positive")
        if limit > self.scientific.top_chunks_per_list:
            raise ValueError(
                f"Search limit cannot exceed {self.scientific.top_chunks_per_list}"
            )
        index_manifest, corpus_manifest = self._validated_loaded_manifests()
        request = {
            "size": limit,
            "track_scores": True,
            "_source": [
                "chunk_id",
                "document_id",
                "ordinal",
                "text",
                "title",
                "corpus",
                "source_url",
                "classification",
                "authored",
                "doc_lang",
            ],
            "query": {"match": {"text": {"query": query, "operator": "or"}}},
            "sort": [
                {"_score": {"order": "desc"}},
                {"chunk_id": {"order": "asc"}},
            ],
        }
        hits = self.transport.search(index_manifest.index_name, request)
        ordered = sorted(
            hits,
            key=lambda hit: (-float(hit.get("_score") or 0.0), str(hit["_id"])),
        )[:limit]
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        index_hash = index_manifest.manifest_hash()
        corpus_hash = corpus_manifest.manifest_hash()
        return [
            RankedChunk(
                chunk_id=str(hit["_source"]["chunk_id"]),
                document_id=str(hit["_source"]["document_id"]),
                ordinal=int(hit["_source"]["ordinal"]),
                text=str(hit["_source"]["text"]),
                title=hit["_source"].get("title"),
                corpus=str(hit["_source"]["corpus"]),
                source_url=hit["_source"].get("source_url"),
                classification=hit["_source"].get("classification"),
                authored=hit["_source"].get("authored"),
                doc_lang=hit["_source"].get("doc_lang"),
                rank=rank,
                score=float(hit.get("_score") or 0.0),
                retriever="bm25",
                query_kind=query_kind,
                query_sha256=query_hash,
                index_manifest_hash=index_hash,
                corpus_manifest_hash=corpus_hash,
            )
            for rank, hit in enumerate(ordered, start=1)
        ]

    def _validate_corpus_manifest(self, manifest: CorpusManifest) -> None:
        if not manifest.coverage_ready:
            raise IncompatibleIndexManifestError(
                "Corpus exclusion coverage is not ready"
            )
        if manifest.scientific_hash != self.scientific.manifest_hash():
            raise IncompatibleIndexManifestError(
                "Corpus scientific manifest does not match retrieval configuration"
            )

    def _validated_loaded_manifests(
        self,
    ) -> tuple[Bm25IndexManifest, CorpusManifest]:
        if self.index_manifest is None or self.corpus_manifest is None:
            raise IncompatibleIndexManifestError(
                "BM25 index has not been built or loaded"
            )
        self._validate_corpus_manifest(self.corpus_manifest)
        if self.index_manifest.scientific_hash != self.scientific.manifest_hash():
            raise IncompatibleIndexManifestError(
                "BM25 scientific manifest does not match retrieval configuration"
            )
        if (
            self.index_manifest.stopwords_hash
            != self.scientific.context_stopwords_hash()
        ):
            raise IncompatibleIndexManifestError(
                "BM25 stop-word manifest does not match retrieval configuration"
            )
        if self.index_manifest.index_definition_hash != index_definition_hash(
            self.scientific
        ):
            raise IncompatibleIndexManifestError(
                "BM25 index definition does not match retrieval configuration"
            )
        if (
            self.index_manifest.corpus_manifest_hash
            != self.corpus_manifest.manifest_hash()
        ):
            raise IncompatibleIndexManifestError(
                "BM25 index manifest does not match the corpus manifest"
            )
        if (
            self.index_manifest.document_count != self.corpus_manifest.document_count
            or self.index_manifest.chunk_count != self.corpus_manifest.chunk_count
        ):
            raise IncompatibleIndexManifestError(
                "BM25 corpus counts do not match the corpus manifest"
            )
        if (
            self.index_manifest.index_name != self.scientific.bm25_index_name
            or self.index_manifest.opensearch_version
            != self.scientific.opensearch_version
        ):
            raise IncompatibleIndexManifestError(
                "BM25 runtime version does not match retrieval configuration"
            )
        return self.index_manifest, self.corpus_manifest


def build_index_definition(scientific: ScientificConfig) -> dict[str, Any]:
    return {
        "settings": {
            "index": {
                "similarity": {
                    "default": {
                        "type": "BM25",
                        "k1": scientific.bm25_k1,
                        "b": scientific.bm25_b,
                    }
                }
            },
            "analysis": {
                "filter": {
                    "visible_english_stop": {
                        "type": "stop",
                        "stopwords": list(scientific.context_stopwords),
                    }
                },
                "analyzer": {
                    "visible_english": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "visible_english_stop"],
                    }
                },
            },
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "chunk_id": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "ordinal": {"type": "integer"},
                "text": {"type": "text", "analyzer": "visible_english"},
                "title": {"type": "keyword", "index": False},
                "corpus": {"type": "keyword"},
                "source_url": {"type": "keyword", "index": False},
                "classification": {"type": "keyword", "index": False},
                "authored": {"type": "keyword", "index": False},
                "doc_lang": {"type": "keyword", "index": False},
            },
        },
    }


def index_definition_hash(scientific: ScientificConfig) -> str:
    return _canonical_hash(build_index_definition(scientific))


class OpenSearchHttpTransport:
    def __init__(self, base_url: str, *, timeout_seconds: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def replace_index(self, index_name: str, definition: dict[str, Any]) -> None:
        self._request("DELETE", f"/{quote(index_name)}", allow_not_found=True)
        self._request("PUT", f"/{quote(index_name)}", json_body=definition)

    def bulk_index(
        self,
        index_name: str,
        documents: Sequence[dict[str, Any]],
    ) -> None:
        lines = []
        for document in documents:
            lines.append(
                json.dumps(
                    {
                        "index": {
                            "_index": index_name,
                            "_id": document["chunk_id"],
                        }
                    },
                    separators=(",", ":"),
                )
            )
            lines.append(json.dumps(document, separators=(",", ":")))
        payload = ("\n".join(lines) + "\n").encode()
        response = self._request(
            "POST",
            "/_bulk",
            data=payload,
            content_type="application/x-ndjson",
        )
        if response.get("errors") is True:
            raise RuntimeError("OpenSearch bulk indexing returned item errors")

    def search(
        self,
        index_name: str,
        request: dict[str, Any],
    ) -> list[dict[str, Any]]:
        response = self._request(
            "POST",
            f"/{quote(index_name)}/_search",
            json_body=request,
        )
        return list(response["hits"]["hits"])

    def refresh(self, index_name: str) -> None:
        self._request("POST", f"/{quote(index_name)}/_refresh")

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        data: bytes | None = None,
        content_type: str = "application/json",
        allow_not_found: bool = False,
    ) -> dict[str, Any]:
        payload = data
        if json_body is not None:
            payload = json.dumps(json_body, separators=(",", ":")).encode()
        request = Request(
            f"{self.base_url}{path}",
            data=payload,
            method=method,
            headers={"Content-Type": content_type},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
        except HTTPError as error:
            if allow_not_found and error.code == 404:
                return {}
            raise RuntimeError(
                f"OpenSearch request failed with status {error.code}"
            ) from error
        return json.loads(body) if body else {}


def _index_document(chunk: CorpusChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "ordinal": chunk.ordinal,
        "text": chunk.text,
        "title": chunk.title,
        "corpus": chunk.corpus,
        "source_url": chunk.source_url,
        "classification": chunk.classification,
        "authored": chunk.authored,
        "doc_lang": chunk.doc_lang,
    }


def _canonical_hash(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()
