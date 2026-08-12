"""Scoring the guardrails against real labelled attacks.

A guardrail with no measured false-positive rate is a liability dressed as a
control. This module produces, for every rail:

    precision, recall, F1, FPR, and the full confusion matrix
    on the test split of the corpus it was fitted on, and
    on a held-out attack corpus it has never seen

The second number is the one that predicts production behaviour. A detector
fitted on one injection corpus and evaluated on the same corpus's test split
measures how well it memorised a phrasing distribution; evaluating it on
Gandalf, written by different people against a different system, measures
whether it learned anything transferable.

Both the heuristic and the trained detector are scored separately and then in
combination, so the contribution of each is visible rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import ARTIFACT_DIR
from ..guardrails.classifier import HashedTextClassifier, model_path
from ..guardrails.detectors import InjectionDetector, JailbreakDetector
from ..security.redaction import Redactor, detect
from .calibration import roc_auc
from .datasets import (
    LabelledSet,
    load_gandalf,
    load_jailbreaks,
    load_pii,
    load_prompt_injections,
)
from .metrics import ClassificationScores, score_classification


@dataclass
class RailReport:
    rail: str
    variant: str
    dataset: str
    scores: ClassificationScores
    auc: float = 0.5
    note: str = ""

    def as_dict(self) -> dict:
        payload = {"rail": self.rail, "variant": self.variant, "dataset": self.dataset,
                   "auc": round(self.auc, 4), **self.scores.as_dict()}
        if self.note:
            payload["note"] = self.note
        return payload


@dataclass
class GuardrailEvaluation:
    reports: list[RailReport] = field(default_factory=list)
    pii: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "training": self.training,
            "rails": [r.as_dict() for r in self.reports],
            "pii": self.pii,
        }

    def get(self, rail: str, variant: str, dataset_contains: str) -> RailReport | None:
        for report in self.reports:
            if (report.rail == rail and report.variant == variant
                    and dataset_contains in report.dataset):
                return report
        return None


# ==========================================================================
# Training
# ==========================================================================

def benign_traffic_sample(n: int = 600) -> list[str]:
    """Real, unambiguously benign questions, for measuring false positives.

    The SQuAD question set is used here as a stand-in for legitimate user
    traffic: 11,873 questions written by crowdworkers with no adversarial
    intent whatsoever. It is not synthetic and it was not written for this
    project, which is what makes it a fair test.

    This exists because of a measured failure. The injection rail was first
    tuned against only the benign class of its own attack corpus, which is
    mostly short topical fragments, and it then rejected 6% of real SQuAD
    questions in the end-to-end run. A rail tuned on one benign distribution
    tells you nothing about its false-positive rate on another, and production
    traffic is always the other one.
    """
    from ..rag.corpus import load_questions, stratified_sample

    try:
        questions = load_questions()
    except FileNotFoundError:
        return []
    return [q.question for q in stratified_sample(questions, n, seed=99)]


def train_detectors(max_fpr: float = 0.05, verbose: bool = True) -> dict:
    """Fit the trained half of each rail on its corpus's train split.

    Threshold selection is a product decision expressed as a constraint:
    maximise recall subject to a false-positive-rate ceiling, measured against
    the union of the corpus's own benign class and a sample of real benign
    user questions. Optimising F1 instead would happily buy a few points of
    recall at a 25% false-positive rate, and a rail that rejects a quarter of
    legitimate questions is a rail that gets switched off within the week.
    """
    summary: dict = {}
    benign = benign_traffic_sample()

    for name, loader, detector_label in (
        ("injection", load_prompt_injections, "prompt injection"),
        ("jailbreak", load_jailbreaks, "jailbreak"),
    ):
        dataset = loader()
        train = dataset.split("train")
        if len(train) < 40:
            summary[name] = {"status": "skipped", "reason": "insufficient training data"}
            continue

        model = HashedTextClassifier(label=detector_label)
        # Weights are fitted on the attack corpus alone. The benign traffic
        # sample deliberately does not go in here: it is the yardstick, and a
        # yardstick you trained on has stopped measuring anything.
        model.fit(train.texts, train.labels)

        tune_texts = list(train.texts) + benign
        tune_labels = list(train.labels) + [0] * len(benign)
        threshold = model.tune_threshold(tune_texts, tune_labels, max_fpr=max_fpr)
        model.save(model_path(name))

        fpr_on_traffic = (
            float((model.score_many(benign) >= threshold).mean()) if benign else None
        )

        summary[name] = {
            "status": "trained",
            "corpus": dataset.name,
            "train_examples": len(train),
            "train_positive": train.n_positive,
            "benign_traffic_examples": len(benign),
            "threshold": round(threshold, 4),
            "max_fpr_constraint": max_fpr,
            "fpr_on_benign_traffic": round(fpr_on_traffic, 4) if fpr_on_traffic is not None else None,
        }
        if verbose:
            extra = f", FPR on real questions {fpr_on_traffic:.3f}" if fpr_on_traffic is not None else ""
            print(f"  trained {name}: {len(train)} examples, threshold {threshold:.3f}{extra}")

    return summary


# ==========================================================================
# Evaluation
# ==========================================================================

def _score_detector(detector, dataset: LabelledSet, use_scores: bool = True):
    verdicts = [detector(text) for text in dataset.texts]
    predictions = [v.fired for v in verdicts]
    labels = [bool(l) for l in dataset.labels]
    scores = score_classification(predictions, labels)
    auc = 0.5
    if use_scores and dataset.n_positive and dataset.n_negative:
        auc = roc_auc(np.array([v.score for v in verdicts]), np.array(dataset.labels))
    return scores, auc


def evaluate_guardrails(train_first: bool = True, verbose: bool = True) -> GuardrailEvaluation:
    evaluation = GuardrailEvaluation()

    if train_first:
        if verbose:
            print("Training detectors on real labelled corpora")
        evaluation.training = train_detectors(verbose=verbose)

    # -- injection ---------------------------------------------------------
    injections = load_prompt_injections()
    test = injections.split("test")

    for variant, detector in (
        ("heuristic", InjectionDetector(use_trained=False)),
        ("trained+heuristic", InjectionDetector(use_trained=True)),
    ):
        scores, auc = _score_detector(detector, test)
        evaluation.reports.append(RailReport(
            rail="injection", variant=variant, dataset=f"{injections.name} [test]",
            scores=scores, auc=auc,
        ))

    # Held out entirely: never fitted on, different authors, different target.
    gandalf = load_gandalf()
    for variant, detector in (
        ("heuristic", InjectionDetector(use_trained=False)),
        ("trained+heuristic", InjectionDetector(use_trained=True)),
    ):
        scores, _ = _score_detector(detector, gandalf, use_scores=False)
        evaluation.reports.append(RailReport(
            rail="injection", variant=variant, dataset=f"{gandalf.name} [held out]",
            scores=scores, auc=0.5,
            note="attack-only corpus: recall is meaningful, FPR is not defined",
        ))

    # -- jailbreak ---------------------------------------------------------
    jailbreaks = load_jailbreaks()
    jb_test = jailbreaks.split("test")
    for variant, detector in (
        ("heuristic", JailbreakDetector(use_trained=False)),
        ("trained+heuristic", JailbreakDetector(use_trained=True)),
    ):
        scores, auc = _score_detector(detector, jb_test)
        evaluation.reports.append(RailReport(
            rail="jailbreak", variant=variant, dataset=f"{jailbreaks.name} [test]",
            scores=scores, auc=auc,
        ))

    # -- PII ---------------------------------------------------------------
    evaluation.pii = evaluate_pii(verbose=verbose)
    return evaluation


def evaluate_pii(verbose: bool = True) -> dict:
    """Span-level recall of the redaction detectors, per entity type.

    A span counts as recalled when a detected span overlaps the annotated one
    and carries the right type. Overlap rather than exact boundaries, because
    redacting "+1 555 0100" when the annotation was "1 555 0100" removes the
    data, which is the outcome the control exists to produce.
    """
    try:
        examples = load_pii()
    except FileNotFoundError:
        return {"status": "unavailable"}

    if not examples:
        return {"status": "no mapped entities in sample"}

    per_type: dict[str, dict[str, int]] = {}
    for example in examples:
        found = detect(example.text, min_confidence=0.5)
        for gold in example.spans:
            bucket = per_type.setdefault(gold["label"], {"total": 0, "hit": 0})
            bucket["total"] += 1
            for span in found:
                overlaps = span.start < gold["end"] and gold["start"] < span.end
                if overlaps and span.entity.value == gold["label"]:
                    bucket["hit"] += 1
                    break

    total = sum(b["total"] for b in per_type.values())
    hit = sum(b["hit"] for b in per_type.values())

    # Also report how much of the corpus's PII the detectors do not even claim
    # to cover. Quoting recall over the mapped subset alone would flatter the
    # rail by hiding the 47 label types it has no detector for.
    from .datasets import PII_LABEL_MAP

    result = {
        "status": "ok",
        "examples": len(examples),
        "mapped_entity_types": sorted(set(PII_LABEL_MAP.values())),
        "overall_recall": round(hit / total, 4) if total else 0.0,
        "spans_evaluated": total,
        "per_entity": {
            entity: {
                "recall": round(b["hit"] / b["total"], 4) if b["total"] else 0.0,
                "n": b["total"],
            }
            for entity, b in sorted(per_type.items())
        },
        "caveat": (
            "Recall is over the entity types these regex detectors target. The "
            "source corpus annotates 56 types; structured identifiers are "
            "detected well and free-text names and addresses are not detected "
            "at all, which is why redaction is treated as defence in depth "
            "rather than as the control that makes egress safe."
        ),
    }

    if verbose:
        print(f"  PII: {result['overall_recall']:.1%} recall over {total} annotated spans")
    return result


def write_report(evaluation: GuardrailEvaluation, path=None):
    import json

    path = path or (ARTIFACT_DIR / "guardrail_eval.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evaluation.as_dict(), indent=2) + "\n")
    return path
