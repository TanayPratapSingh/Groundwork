"""Load the real corpus and the real question set.

The corpus is the 1,204 Wikipedia paragraphs in the SQuAD v2.0 dev set. The
questions are the 11,873 written by SQuAD's crowdworkers against those
paragraphs, with human-marked gold answer spans.

Why this dataset rather than a corpus assembled for the demo: 5,945 of those
questions are *deliberately unanswerable*. Crowdworkers wrote them to look
answerable from the paragraph while having no answer in it. That gives the
abstain policy and the calibration study real labels, which is the part of a
RAG evaluation that is otherwise almost always faked.

The retrieval labels are free and exact: the gold passage for a question is
the paragraph the crowdworker was looking at when they wrote it.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from ..config import RAW_DIR

SQUAD_PATH = RAW_DIR / "squad_v2_dev.json"


class DatasetMissing(FileNotFoundError):
    """Raised with the exact command that fixes it."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"{path} not found. This repo ships no data; fetch the real "
            f"datasets first:\n\n    python3 scripts/fetch_data.py\n"
        )


@dataclass(frozen=True)
class Document:
    """One retrievable passage."""

    doc_id: str
    title: str
    text: str

    @property
    def n_chars(self) -> int:
        return len(self.text)


@dataclass
class Question:
    """One evaluation item, with its real labels."""

    qid: str
    question: str
    gold_doc_id: str
    title: str
    # Every distinct human answer string. Empty for unanswerable questions.
    answers: list[str] = field(default_factory=list)
    is_impossible: bool = False

    @property
    def answerable(self) -> bool:
        return not self.is_impossible


def _doc_id(title: str, index: int, text: str) -> str:
    """Stable ID: title, ordinal, and a hash of the text.

    The hash means a doc ID changes if and only if its content changes, so a
    stale index is detectable rather than silently mismatched against fresh
    labels.
    """
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).hexdigest()
    slug = title.replace(" ", "_")
    return f"{slug}#{index:03d}:{digest}"


def _load_raw(path: Path | None = None) -> dict:
    path = path or SQUAD_PATH
    if not path.exists():
        raise DatasetMissing(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_corpus(path: Path | None = None) -> list[Document]:
    """Every paragraph, deduplicated, in a stable order."""
    payload = _load_raw(path)
    docs: list[Document] = []
    seen: set[str] = set()

    for article in payload["data"]:
        title = article["title"].replace("_", " ")
        for index, paragraph in enumerate(article["paragraphs"]):
            text = paragraph["context"].strip()
            if not text:
                continue
            doc_id = _doc_id(article["title"], index, text)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            docs.append(Document(doc_id=doc_id, title=title, text=text))
    return docs


def load_questions(path: Path | None = None) -> list[Question]:
    """Every question, carrying its gold passage and gold answer strings."""
    payload = _load_raw(path)
    questions: list[Question] = []

    for article in payload["data"]:
        title = article["title"].replace("_", " ")
        for index, paragraph in enumerate(article["paragraphs"]):
            text = paragraph["context"].strip()
            if not text:
                continue
            doc_id = _doc_id(article["title"], index, text)
            for qa in paragraph["qas"]:
                # Deduplicate while preserving order: SQuAD collects several
                # independent annotations per question and they often agree.
                answers, seen_ans = [], set()
                for a in qa.get("answers", []):
                    norm = a["text"].strip()
                    if norm and norm.lower() not in seen_ans:
                        seen_ans.add(norm.lower())
                        answers.append(norm)
                questions.append(Question(
                    qid=qa["id"],
                    question=qa["question"].strip(),
                    gold_doc_id=doc_id,
                    title=title,
                    answers=answers,
                    is_impossible=bool(qa.get("is_impossible", False)),
                ))
    return questions


def stratified_sample(
    questions: list[Question],
    n: int,
    seed: int = 20260811,
    balance: bool = True,
) -> list[Question]:
    """A reproducible sample that preserves the answerable/unanswerable mix.

    Full-set evaluation is 11,873 questions and takes minutes; a sample keeps
    the inner development loop fast. The seed is fixed and the split is
    stratified so a sampled run is comparable across commits, which is the only
    way a regression gate on a sample means anything.
    """
    if n >= len(questions):
        return list(questions)

    rng = random.Random(seed)
    if not balance:
        return rng.sample(questions, n)

    answerable = [q for q in questions if q.answerable]
    impossible = [q for q in questions if not q.answerable]
    share = len(answerable) / len(questions) if questions else 0.5
    n_ans = min(len(answerable), round(n * share))
    n_imp = min(len(impossible), n - n_ans)

    picked = rng.sample(answerable, n_ans) + rng.sample(impossible, n_imp)
    rng.shuffle(picked)
    return picked


def split_questions(
    questions: list[Question],
    fit_fraction: float = 0.5,
    seed: int = 20260811,
) -> tuple[list[Question], list[Question]]:
    """Deterministic fit/test split for calibration.

    Split by hashing the question ID rather than by shuffling, so the
    assignment of any given question is stable no matter what else is in the
    list. That prevents the quiet leak where growing the eval set reshuffles
    which items were used to fit the temperature.
    """
    fit, test = [], []
    for q in questions:
        h = hashlib.blake2b(f"{seed}:{q.qid}".encode(), digest_size=4).hexdigest()
        bucket = int(h, 16) / 0xFFFFFFFF
        (fit if bucket < fit_fraction else test).append(q)
    return fit, test


def corpus_stats(docs: list[Document], questions: list[Question]) -> dict:
    lengths = [d.n_chars for d in docs]
    return {
        "documents": len(docs),
        "titles": len({d.title for d in docs}),
        "mean_doc_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "questions": len(questions),
        "answerable": sum(1 for q in questions if q.answerable),
        "unanswerable": sum(1 for q in questions if not q.answerable),
    }
