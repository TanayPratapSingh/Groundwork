"""The evaluation harness and the release gate.

Four stages, each answering a different question:

    retrieval    can the retriever find the right passage at all?
    end to end   does the whole pipeline produce the right answer, and does it
                 decline when there is no right answer to produce?
    calibration  when it says 0.8, is it right 80 percent of the time?
    gate         is this build allowed to ship?

The separation is the point. When end-to-end quality drops, the retrieval
numbers tell you immediately whether the retriever regressed or the generation
step did, and without that split every regression investigation starts from
zero.

Everything is computed against real labels: the gold passage is the paragraph
each SQuAD question was written from, the gold answers are the crowdworkers',
and the unanswerable flag is theirs too. Nothing in this file invents an
expected output.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import ARTIFACT_DIR, DEFAULT_TIER, SETTINGS
from ..rag.confidence import ConfidenceModel, features_to_vector
from ..rag.corpus import Question, load_corpus, load_questions, split_questions, stratified_sample
from ..rag.index import HybridIndex
from ..rag.pipeline import Decision, RagPipeline
from ..rag.retriever import Retriever
from .calibration import CalibrationReport, evaluate_calibration, sweep_thresholds
from .metrics import (
    ClassificationScores,
    RetrievalScores,
    contains_gold,
    exact_match,
    score_classification,
    score_retrieval,
    token_f1,
)


# ==========================================================================
# Stage 1: retrieval
# ==========================================================================

def evaluate_retrieval(
    index: HybridIndex,
    questions: list[Question],
    modes: tuple[str, ...] = ("hybrid", "dense", "sparse"),
    depth: int = 10,
) -> dict[str, RetrievalScores]:
    """Retrieval quality per mode, with no model calls at all.

    This runs in seconds over thousands of questions, which is what makes it
    usable as a pre-commit check. Reporting all three modes turns "hybrid
    retrieval is better" from a design claim into a measurement, and if the
    fusion ever stops earning its complexity, this is where it shows up.
    """
    retriever = Retriever(index)
    results: dict[str, RetrievalScores] = {}

    for mode in modes:
        rankings = [
            (retriever.rank_all(q.question, mode=mode, depth=depth), q.gold_doc_id)
            for q in questions
        ]
        results[mode] = score_retrieval(rankings)
    return results


# ==========================================================================
# Stage 2: end to end
# ==========================================================================

@dataclass
class ItemResult:
    """One question's outcome, recorded *before* the abstain threshold applies.

    `candidate_correct` is the key field: it records whether the answer the
    pipeline produced would have been right, independent of whether the
    confidence policy chose to return it. That separation is what lets the
    harness fit a confidence model and then pick an operating point, instead
    of evaluating a policy on data the same policy already filtered.
    """

    qid: str
    question: str
    answerable: bool
    # True when generation happened at all: rails passed and retrieval was
    # non-empty. False means the request was stopped before any candidate
    # existed, and no threshold can rescue it.
    has_candidate: bool
    confidence: float
    groundedness: float
    candidate_correct: bool
    gold_retrieved: bool
    exact_match: float
    token_f1: float
    stopped_by: str = ""
    features: dict = field(default_factory=dict)
    latency_ms: float = 0.0

    def answered_at(self, threshold: float) -> bool:
        return self.has_candidate and self.confidence >= threshold

    def correct_at(self, threshold: float) -> bool:
        """Correctness of the *decision*, which depends on answerability.

        answerable    correct means we answered and the gold span is present.
                      Declining is wrong: the answer existed and we withheld it.
        unanswerable  correct means we declined. A fluent answer to a question
                      with no answer is the failure this project exists to
                      measure, and scoring it as merely 'low F1' understates it.
        """
        answered = self.answered_at(threshold)
        if self.answerable:
            return answered and self.candidate_correct
        return not answered

    def as_dict(self) -> dict:
        return {
            "qid": self.qid,
            "answerable": self.answerable,
            "has_candidate": self.has_candidate,
            "confidence": round(self.confidence, 4),
            "candidate_correct": self.candidate_correct,
            "gold_retrieved": self.gold_retrieved,
            "stopped_by": self.stopped_by,
        }


def run_pipeline_eval(
    pipeline: RagPipeline,
    questions: list[Question],
    tenant: str = "evaluation",
    tier: str = DEFAULT_TIER,
    progress_every: int = 250,
    verbose: bool = True,
) -> list[ItemResult]:
    """Run every question, capturing the candidate answer and its features.

    `apply_decision=False` so the threshold is not applied here. The pipeline
    still computes a confidence; the harness decides later what to do with it.
    """
    results: list[ItemResult] = []

    for index, question in enumerate(questions, start=1):
        answer = pipeline.answer(
            question.question, tenant=tenant, tier=tier, apply_decision=False
        )
        candidate = answer.candidate_text
        has_candidate = bool(candidate)
        gold_retrieved = question.gold_doc_id in [p.doc_id for p in answer.passages]

        correct = bool(has_candidate and contains_gold(candidate, question.answers)) \
            if question.answerable else False
        em = exact_match(candidate, question.answers) if has_candidate and question.answerable else 0.0
        f1 = token_f1(candidate, question.answers) if has_candidate and question.answerable else 0.0

        results.append(ItemResult(
            qid=question.qid,
            question=question.question,
            answerable=question.answerable,
            has_candidate=has_candidate,
            confidence=answer.confidence,
            groundedness=answer.groundedness,
            candidate_correct=correct,
            gold_retrieved=gold_retrieved,
            exact_match=em,
            token_f1=f1,
            stopped_by=",".join(answer.blocked_by) if not has_candidate else "",
            features=answer.features,
            latency_ms=answer.latency_ms,
        ))

        if verbose and progress_every and index % progress_every == 0:
            print(f"    {index}/{len(questions)} processed")

    return results


def choose_threshold(
    results: list[ItemResult],
    min_precision: float | None = None,
    steps: int = 101,
) -> tuple[float, dict]:
    """Pick the answer/abstain cutoff: maximise coverage subject to precision.

    Fitted on the fit split only.

    The obvious objective, maximising decision accuracy, was tried first and
    degenerates. SQuAD v2 is almost exactly half unanswerable, so a system that
    answers nothing already scores ~0.50, and with a confidence signal at AUC
    0.61 the accuracy-optimal cutoff sits high enough to abstain on nearly
    everything: it chose 0.48 and delivered correct answers to 7% of answerable
    questions. That is the true optimum for the metric and a useless product.

    The objective here is the one a product is actually specified with: *answer
    as much as possible, provided that when we answer we are right at least
    `min_precision` of the time.* Precision counts answering an unanswerable
    question as wrong, so the guarantee covers confabulation rather than just
    accuracy on the easy half.

    If no threshold reaches the precision floor, the highest-precision point is
    returned and flagged, because silently falling back to a promise the system
    cannot keep is worse than reporting that it cannot keep it.
    """
    if not results:
        return 0.5, {"threshold": 0.5, "fit_n": 0, "status": "no data"}

    floor = min_precision if min_precision is not None else SETTINGS.decision.min_answer_precision
    rows = []
    for step in range(steps):
        threshold = step / (steps - 1)
        answered = [r for r in results if r.answered_at(threshold)]
        correct = sum(r.correct_at(threshold) for r in answered)
        precision = correct / len(answered) if answered else 1.0
        rows.append({
            "threshold": threshold,
            "coverage": len(answered) / len(results),
            "precision": precision,
            "correct_answers": correct,
        })

    feasible = [row for row in rows if row["precision"] >= floor and row["coverage"] > 0]
    if feasible:
        chosen = max(feasible, key=lambda row: (row["coverage"], -row["threshold"]))
        status = "precision floor met"
    else:
        chosen = max(rows, key=lambda row: (row["precision"], row["coverage"]))
        status = f"precision floor {floor:.2f} unreachable; using best available"

    return chosen["threshold"], {
        "threshold": round(chosen["threshold"], 4),
        "objective": f"max coverage subject to precision >= {floor:.2f}",
        "status": status,
        "fit_coverage": round(chosen["coverage"], 4),
        "fit_precision": round(chosen["precision"], 4),
        "fit_n": len(results),
    }


def summarise(results: list[ItemResult], threshold: float) -> dict:
    """Headline numbers, split by answerable and unanswerable.

    Reported separately on purpose. A single blended accuracy hides the tradeoff
    that defines the system: you can move answer rate and abstention rate in
    opposite directions all day and keep the blended number flat.
    """
    if not results:
        return {}

    answerable = [r for r in results if r.answerable]
    impossible = [r for r in results if not r.answerable]

    def mean(values) -> float:
        values = list(values)
        return float(np.mean(values)) if values else 0.0

    # Abstention treated as binary classification: positive = should decline.
    abstention = score_classification(
        predictions=[not r.answered_at(threshold) for r in results],
        labels=[not r.answerable for r in results],
    )

    # Ceiling analysis. `candidate_correct` over answerable questions is what
    # the system would score with a perfect decision policy, so the gap
    # between it and containment is exactly the cost of the threshold. Without
    # this number, a drop in containment is ambiguous between "generation got
    # worse" and "the policy got more cautious".
    ceiling = mean(r.candidate_correct for r in answerable)

    return {
        "n": len(results),
        "operating_threshold": round(threshold, 4),
        "overall_accuracy": round(mean(r.correct_at(threshold) for r in results), 4),
        "answerable": {
            "n": len(answerable),
            "containment": round(mean(r.correct_at(threshold) for r in answerable), 4),
            "containment_ceiling": round(ceiling, 4),
            "exact_match": round(mean(r.exact_match for r in answerable), 4),
            "token_f1": round(mean(r.token_f1 for r in answerable), 4),
            "answer_rate": round(mean(r.answered_at(threshold) for r in answerable), 4),
            "gold_retrieved": round(mean(r.gold_retrieved for r in answerable), 4),
            "generation_available": round(mean(r.has_candidate for r in answerable), 4),
        },
        "unanswerable": {
            "n": len(impossible),
            "correct_abstention": round(mean(r.correct_at(threshold) for r in impossible), 4),
            "false_answer_rate": round(mean(r.answered_at(threshold) for r in impossible), 4),
        },
        "abstention_classifier": abstention.as_dict(),
        "stopped_before_generation": {
            reason: sum(1 for r in results if r.stopped_by == reason)
            for reason in sorted({r.stopped_by for r in results if r.stopped_by})
        },
        "mean_groundedness": round(mean(r.groundedness for r in results if r.has_candidate), 4),
        "mean_latency_ms": round(mean(r.latency_ms for r in results), 2),
    }


# ==========================================================================
# Stage 3: calibration
# ==========================================================================

def _matrix(results: list[ItemResult]):
    usable = [r for r in results if r.features]
    if not usable:
        return np.zeros((0, 6)), np.zeros(0), []
    X = np.vstack([features_to_vector(r.features) for r in usable])
    # The target is "would answering right now be correct". For an
    # unanswerable question that is always false, which is what teaches the
    # model to be unconfident exactly where confidence is dangerous.
    y = np.array([1.0 if r.candidate_correct else 0.0 for r in usable])
    return X, y, usable


def fit_and_evaluate_confidence(
    fit_results: list[ItemResult],
    test_results: list[ItemResult],
    save: bool = True,
) -> tuple[ConfidenceModel, CalibrationReport, CalibrationReport, list[dict]]:
    """Fit the confidence model on one split, report honestly on the other.

    Both the weights and the temperature come from `fit_results` and never
    touch `test_results`. Fitting a temperature on the same data you then
    report ECE from produces a beautiful number that means nothing, and it is
    the most common way calibration results get overstated.
    """
    X_fit, y_fit, _ = _matrix(fit_results)
    X_test, y_test, _ = _matrix(test_results)

    model = ConfidenceModel()
    if X_fit.shape[0] >= 50:
        model.fit(X_fit, y_fit)

    before = evaluate_calibration(model.predict_proba(X_test), y_test) if X_test.size \
        else evaluate_calibration([], [])

    if X_fit.shape[0] >= 50:
        # Temperature is fitted on a held-out slice of the fit split, so it
        # never sees the weights' own training data or the test split.
        half = max(1, X_fit.shape[0] // 2)
        model.fit_temperature(X_fit[half:], y_fit[half:])

    after = evaluate_calibration(
        model.predict_proba(X_test), y_test, temperature=model.temperature
    ) if X_test.size else before

    sweep = sweep_thresholds(model.predict_proba(X_test), y_test) if X_test.size else []

    if save:
        model.save()
    return model, before, after, sweep


def rescore(results: list[ItemResult], model: ConfidenceModel) -> None:
    """Recompute confidences in place with the fitted model.

    Necessary and easy to forget. The confidences recorded during the run came
    from the prior model that existed before fitting, so scoring the decision
    policy against them would measure the prior, not the model that ships.
    """
    for result in results:
        if result.features:
            result.confidence = model.predict_one(result.features)
        else:
            result.confidence = 0.0


# ==========================================================================
# Stage 4: the gate
# ==========================================================================

@dataclass
class GateResult:
    passed: bool
    checks: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"passed": self.passed, "checks": self.checks}

    def render(self) -> str:
        lines = []
        for check in self.checks:
            if check.get("skipped"):
                lines.append(f"  [SKIP] {check['metric']:<28} {check['skipped']}")
                continue
            mark = "PASS" if check["passed"] else "FAIL"
            lines.append(
                f"  [{mark}] {check['metric']:<28} {check['value']:.4f} "
                f"{check['comparator']} {check['threshold']:.4f}"
            )
        lines.append(f"\n  Gate: {'PASSED' if self.passed else 'FAILED'}")
        return "\n".join(lines)


def check_gate(
    retrieval: dict[str, RetrievalScores],
    summary: dict,
    calibration: CalibrationReport,
    guardrails: dict | None = None,
    baseline: dict | None = None,
    sample_size: int | None = None,
) -> GateResult:
    """Absolute floors, plus a regression check against the previous release.

    Both are needed. Floors alone let a build that was always mediocre keep
    shipping; regression checks alone let quality slide indefinitely, two
    percent at a time, with every individual release passing.
    """
    gate = SETTINGS.gate
    checks: list[dict] = []

    def add(metric: str, value: float, threshold: float, comparator: str) -> None:
        passed = value >= threshold if comparator == ">=" else value <= threshold
        checks.append({
            "metric": metric, "value": float(value), "threshold": float(threshold),
            "comparator": comparator, "passed": bool(passed),
        })

    # Gate on the mode actually in production, not on whichever mode sounds
    # most impressive. These diverged during development: the gate was reading
    # `hybrid` (recall@5 0.851) while the pipeline had already been switched to
    # `sparse` (0.908) on the strength of the ablation, so the gate was
    # protecting a code path no request took.
    active = retrieval.get(SETTINGS.retrieval.default_mode) or retrieval.get("hybrid")
    if active:
        add("retrieval.recall@5", active.recall_at_5, gate.min_recall_at_5, ">=")
        add("retrieval.mrr", active.mrr, gate.min_mrr, ">=")

    if summary:
        add("answer.containment", summary["answerable"]["containment"],
            gate.min_answer_containment, ">=")
        add("abstention.recall", summary["abstention_classifier"]["recall"],
            gate.min_abstention_recall, ">=")

    add("calibration.ece", calibration.ece, gate.max_ece, "<=")

    if guardrails:
        for report in guardrails.get("rails", []):
            if report["rail"] == "injection" and report["variant"] == "trained+heuristic" \
                    and "[test]" in report["dataset"]:
                add("guardrail.injection_recall", report["recall"],
                    gate.min_injection_recall, ">=")
                add("guardrail.injection_fpr", report["fpr"], gate.max_injection_fpr, "<=")

    # Regression check. Only metrics present in both runs are compared, so
    # adding a new metric never fails the first build that introduces it.
    #
    # And only when the two runs evaluated the same number of questions.
    # Comparing a 400-question run against a 1,200-question baseline compares
    # sampling noise: ECE in particular moved from 0.044 to 0.172 purely on
    # sample size, which would fail every small run and train people to pass
    # --no-guardrails or ignore the gate. Different sample size means the runs
    # are not comparable, not that quality regressed.
    if baseline and baseline.get("n") not in (None, sample_size):
        checks.append({
            "metric": "regression check", "value": float(sample_size),
            "threshold": float(baseline.get("n") or 0), "comparator": "==",
            "passed": True, "skipped": "sample size differs from baseline",
        })
        baseline = None

    if baseline:
        for check in list(checks):
            previous = baseline.get(check["metric"])
            if previous is None:
                continue
            higher_is_better = check["comparator"] == ">="
            delta = check["value"] - previous
            regressed = (delta < -gate.max_regression) if higher_is_better \
                else (delta > gate.max_regression)
            if regressed:
                checks.append({
                    "metric": f"{check['metric']} (vs baseline)",
                    "value": float(check["value"]), "threshold": float(previous),
                    "comparator": check["comparator"], "passed": False,
                })

    return GateResult(passed=all(c["passed"] for c in checks), checks=checks)


# ==========================================================================
# Orchestration
# ==========================================================================

@dataclass
class EvaluationRun:
    started_at: str
    provider: str
    corpus: dict
    retrieval: dict
    summary: dict
    calibration_before: dict
    calibration_after: dict
    threshold_sweep: list[dict]
    confidence_model: dict
    guardrails: dict
    gate: dict
    duration_s: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def full_evaluation(
    n_questions: int = 1200,
    include_guardrails: bool = True,
    baseline_path: Path | None = None,
    verbose: bool = True,
) -> EvaluationRun:
    started = time.perf_counter()

    if verbose:
        print("Loading real corpus (SQuAD v2.0 dev)")
    documents = load_corpus()
    questions = load_questions()

    index = HybridIndex()
    try:
        index = HybridIndex.load()
        if len(index) != len(documents):
            raise FileNotFoundError("stale index")
    except FileNotFoundError:
        if verbose:
            print(f"  building index over {len(documents)} passages")
        index.build(documents)
        index.save()

    sample = stratified_sample(questions, n_questions)
    if verbose:
        print(f"  {len(documents)} passages, evaluating {len(sample)} of {len(questions)} questions")

    if verbose:
        print("\nStage 1: retrieval")
    retrieval = evaluate_retrieval(index, sample)
    if verbose:
        for mode, scores in retrieval.items():
            print(f"  {mode:<8} recall@5 {scores.recall_at_5:.3f}  MRR {scores.mrr:.3f}")

    guardrails: dict = {}
    if include_guardrails:
        if verbose:
            print("\nStage 2: guardrails")
        from .guardrail_eval import evaluate_guardrails
        guardrails = evaluate_guardrails(train_first=True, verbose=verbose).as_dict()

    if verbose:
        print("\nStage 3: end to end")
    pipeline = RagPipeline(index)
    results = run_pipeline_eval(pipeline, sample, verbose=verbose)

    if verbose:
        print("\nStage 4: calibration and operating point")
    fit_ids = {q.qid for q in split_questions(sample)[0]}
    fit_results = [r for r in results if r.qid in fit_ids]
    test_results = [r for r in results if r.qid not in fit_ids]

    model, before, after, sweep = fit_and_evaluate_confidence(fit_results, test_results)

    # The confidences recorded during the run came from the unfitted prior.
    # Replace them before any decision is scored.
    rescore(results, model)

    threshold, threshold_info = choose_threshold(fit_results)
    if verbose:
        print(f"  ECE {before.ece:.4f} -> {after.ece:.4f} after temperature scaling "
              f"(T={model.temperature:.3f}), AUC {after.auc:.4f}")
        print(f"  operating threshold {threshold:.2f} "
              f"(chosen on {threshold_info['fit_n']} fit-split questions)")

    # Reported on the test split only: the threshold was chosen on the fit
    # split, so scoring it on the fit split would report its own training fit.
    summary = summarise(test_results, threshold)
    summary["threshold_selection"] = threshold_info

    baseline = None
    baseline_path = baseline_path or (ARTIFACT_DIR / "baseline.json")
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text()).get("metrics")

    gate = check_gate(retrieval, summary, after, guardrails, baseline,
                      sample_size=len(sample))
    if verbose:
        print("\nRelease gate")
        print(gate.render())

    return EvaluationRun(
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        provider=SETTINGS.provider,
        corpus={
            "documents": len(documents),
            "questions_total": len(questions),
            "questions_evaluated": len(sample),
            "index": index.stats(),
        },
        retrieval={mode: scores.as_dict() for mode, scores in retrieval.items()},
        summary=summary,
        calibration_before=before.as_dict(),
        calibration_after=after.as_dict(),
        threshold_sweep=sweep,
        confidence_model={
            "coefficients": model.coefficients(),
            "bias": round(model.bias, 4),
            "temperature": round(model.temperature, 4),
            "n_train": model.n_train,
        },
        guardrails=guardrails,
        gate=gate.as_dict(),
        duration_s=round(time.perf_counter() - started, 2),
    )
