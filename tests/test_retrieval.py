"""Retrieval: index, BM25, fusion, and the context block the model sees."""

from __future__ import annotations

import pytest

from conftest import requires_corpus
from groundwork.rag.corpus import load_corpus, stratified_sample
from groundwork.rag.embed import HashingEmbedder, tokenize
from groundwork.rag.index import HybridIndex
from groundwork.rag.retriever import Retriever


def test_tokenizer_drops_stopwords_and_lowercases():
    assert tokenize("The Rhine flows THROUGH Germany") == ["rhine", "flows", "through", "germany"]


def test_embeddings_are_unit_length():
    embedder = HashingEmbedder(dim=128)
    vector = embedder.embed_one("photosynthesis converts light energy")
    assert vector.shape == (128,)
    assert abs(float((vector @ vector) ** 0.5) - 1.0) < 1e-5


def test_empty_text_embeds_to_zero_without_crashing():
    # Guards a real edge case: an all-stopword query must not divide by zero.
    assert float(HashingEmbedder(dim=64).embed_one("the of and").sum()) == 0.0


def test_retrieval_finds_the_relevant_passage(tiny_index):
    result = Retriever(tiny_index).retrieve("What is the largest inland port in Europe?")
    assert result.passages
    assert result.passages[0].doc_id == "doc-3"


def test_citation_tags_are_sequential(tiny_index):
    result = Retriever(tiny_index).retrieve("energy")
    assert [p.citation_tag for p in result.passages] == [
        f"S{i}" for i in range(1, len(result.passages) + 1)
    ]


def test_context_block_respects_its_character_budget(tiny_index):
    result = Retriever(tiny_index).retrieve("energy")
    assert len(result.context_block(max_chars=200)) <= 400  # one passage may overshoot the budget


def test_nonsense_query_retrieves_nothing(tiny_index):
    result = Retriever(tiny_index).retrieve("zzzz qqqq xxxx")
    assert result.empty and not result.passages


def test_retrieval_modes_are_independent(tiny_index):
    retriever = Retriever(tiny_index)
    for mode in ("hybrid", "dense", "sparse"):
        assert retriever.retrieve("chlorophyll", mode=mode) is not None


def test_index_round_trips_through_disk(tiny_index, tmp_path):
    vectors, meta = tmp_path / "v.npz", tmp_path / "m.json"
    tiny_index.save(vectors, meta)
    reloaded = HybridIndex.load(vectors, meta)

    assert len(reloaded) == len(tiny_index)
    before = Retriever(tiny_index).retrieve("inland port")
    after = Retriever(reloaded).retrieve("inland port")
    assert [p.doc_id for p in before.passages] == [p.doc_id for p in after.passages]


def test_missing_index_names_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_index"):
        HybridIndex.load(tmp_path / "nope.npz", tmp_path / "nope.json")


@requires_corpus
def test_document_ids_are_unique_across_the_real_corpus():
    documents = load_corpus()
    assert len(documents) > 1000
    assert len({d.doc_id for d in documents}) == len(documents)


@requires_corpus
def test_stratified_sample_is_deterministic_and_balanced():
    from groundwork.rag.corpus import load_questions

    questions = load_questions()
    first = stratified_sample(questions, 200)
    second = stratified_sample(questions, 200)
    assert [q.qid for q in first] == [q.qid for q in second]

    answerable = sum(1 for q in first if q.answerable)
    assert 60 <= answerable <= 140  # roughly the corpus's own 50/50 mix
