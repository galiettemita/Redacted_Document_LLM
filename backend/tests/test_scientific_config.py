from redacted_backend.scientific_config import ScientificConfig


def test_scientific_manifest_is_stable_and_contains_disclosed_parameters() -> None:
    config = ScientificConfig()

    assert config.model_dump() == {
        "chunk_words": 300,
        "chunk_overlap": 0,
        "bm25_k1": 1.2,
        "bm25_b": 0.75,
        "top_chunks_per_list": 300,
        "rrf_k": 60,
        "top_documents": 100,
        "chunks_per_document": 2,
        "pdf_render_dpi": 144,
        "visible_transcript_version": "visible-transcript-v1",
        "mask_detection_version": "mask-discovery-v1",
        "length_cue_version": "visible-geometry-v1",
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "model": "gpt-6-astra",
        "reasoning_effort": "medium",
    }
    assert len(config.manifest_hash()) == 64
    assert config.manifest_hash() == ScientificConfig().manifest_hash()
