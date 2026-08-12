"""Evaluation: metrics, calibration, guardrail scoring, and the release gate."""

from .calibration import (
    CalibrationReport,
    evaluate_calibration,
    expected_calibration_error,
    roc_auc,
    sweep_thresholds,
)
from .datasets import load_gandalf, load_jailbreaks, load_pii, load_prompt_injections
from .guardrail_eval import evaluate_guardrails, train_detectors
from .metrics import (
    ClassificationScores,
    RetrievalScores,
    contains_gold,
    exact_match,
    normalize_answer,
    score_classification,
    score_retrieval,
    token_f1,
)
from .runner import check_gate, evaluate_retrieval, full_evaluation, run_pipeline_eval, summarise

__all__ = [
    "CalibrationReport",
    "ClassificationScores",
    "RetrievalScores",
    "check_gate",
    "contains_gold",
    "evaluate_calibration",
    "evaluate_guardrails",
    "evaluate_retrieval",
    "exact_match",
    "expected_calibration_error",
    "full_evaluation",
    "load_gandalf",
    "load_jailbreaks",
    "load_pii",
    "load_prompt_injections",
    "normalize_answer",
    "roc_auc",
    "run_pipeline_eval",
    "score_classification",
    "score_retrieval",
    "summarise",
    "sweep_thresholds",
    "token_f1",
    "train_detectors",
]
