from __future__ import annotations

from redacted_backend.corpus.chunker import CorpusDocument, chunk_document


def corpus_document(*, doc_id: str = "doc-1", body: str) -> CorpusDocument:
    return CorpusDocument(
        doc_id=doc_id,
        classification="SECRET",
        authored="Visible author",
        title="Visible title",
        body=body,
        corpus="cfpf",
        source_url="https://example.test/source",
        pg_cnt=2,
        char_cnt=len(body),
        word_cnt=len(body.split()),
        doc_lang="en",
    )


def words(count: int) -> str:
    return " ".join(f"w{index}" for index in range(count))


def test_a_300_word_paragraph_remains_one_chunk() -> None:
    chunks = chunk_document(corpus_document(body=words(300)))

    assert [chunk.word_count for chunk in chunks] == [300]
    assert chunks[0].text == words(300)


def test_a_301_word_unsentenced_paragraph_splits_without_overlap() -> None:
    chunks = chunk_document(corpus_document(body=words(301)))

    assert [chunk.word_count for chunk in chunks] == [300, 1]
    assert " ".join(chunk.text for chunk in chunks).split() == words(301).split()


def test_long_paragraph_prefers_sentence_boundaries_before_word_boundaries() -> None:
    first_sentence = f"{words(180)}."
    second_sentence = f"{words(121)}."

    chunks = chunk_document(corpus_document(body=f"{first_sentence} {second_sentence}"))

    assert [chunk.word_count for chunk in chunks] == [180, 121]


def test_chunk_ids_and_normalized_text_are_deterministic() -> None:
    original = corpus_document(body="Alpha   beta.\n\nGamma delta.")

    first = chunk_document(original)
    second = chunk_document(original)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert (
        first[0].id
        == "d4c98e20d905e4ebf24c1376f9fdbb9f20a409745ec4be6c6a41dacf254e7060"
    )
    assert first[0].text == "Alpha beta.\n\nGamma delta."
