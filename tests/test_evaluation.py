"""Evaluation: metrics, calibration, drift, and the release gate."""

from __future__ import annotations

import numpy as np
import pytest

from groundwork.evaluation.calibration import (
    brier_score,
    evaluate_calibration,
    expected_calibration_error,
    roc_auc,
    sweep_thresholds,
)
from groundwork.evaluation.metrics import (
    contains_gold,
    exact_match,
    ndcg_at_k,
    normalize_answer,
    recall_at_k,
    reciprocal_rank,
    score_classification,
    score_retrieval,
    token_f1,
)
from groundwork.telemetry.drift import DriftMonitor, ks_statistic, population_stability_index


# -- retrieval metrics -----------------------------------------------------

def test_recall_at_k_respects_the_cutoff():
    ranked = ["a", "b", "c", "d", "e"]
    assert recall_at_k(ranked, "c", 3) == 1.0
    assert recall_at_k(ranked, "e", 3) == 0.0


def test_reciprocal_rank_rewards_position():
    assert reciprocal_rank(["a", "b", "c"], "a") == 1.0
    assert reciprocal_rank(["a", "b", "c"], "c") == pytest.approx(1 / 3)
    assert reciprocal_rank(["a", "b"], "z") == 0.0


def test_ndcg_discounts_lower_ranks():
    assert ndcg_at_k(["a"], "a", 10) == 1.0
    assert ndcg_at_k(["x", "a"], "a", 10) < 1.0


def test_score_retrieval_aggregates():
    scores = score_retrieval([(["a", "b"], "a"), (["x", "y"], "y")])
    assert scores.n == 2
    assert scores.recall_at_3 == 1.0
    assert scores.mrr == pytest.approx(0.75)


# -- answer metrics --------------------------------------------------------

def test_squad_normalisation_matches_the_official_rules():
    assert normalize_answer("The  Rhine, ") == "rhine"
    assert normalize_answer("A cat!") == "cat"


def test_exact_match_uses_normalisation():
    assert exact_match("The Rhine", ["Rhine"]) == 1.0
    assert exact_match("Danube", ["Rhine"]) == 0.0


def test_token_f1_partially_credits_overlap():
    assert 0.0 < token_f1("the Rhine river", ["Rhine"]) < 1.0


def test_containment_credits_a_sentence_holding_the_span():
    """The metric that matters for a cited-sentence answer: SQuAD gold spans
    are minimal, so token F1 punishes a correct sentence for being a sentence."""
    assert contains_gold("Duisburg has Europe's largest inland port [S1].", ["Duisburg"]) == 1.0
    assert contains_gold("Rotterdam is a port city.", ["Duisburg"]) == 0.0


# -- classification --------------------------------------------------------

def test_classification_scores():
    scores = score_classification([True, True, False, False], [True, False, True, False])
    assert scores.tp == 1 and scores.fp == 1 and scores.fn == 1 and scores.tn == 1
    assert scores.precision == 0.5 and scores.recall == 0.5
    assert scores.false_positive_rate == 0.5


# -- calibration -----------------------------------------------------------

def test_perfect_calibration_has_near_zero_ece():
    confidences = np.array([0.1] * 100 + [0.9] * 100)
    correct = np.array([0.0] * 90 + [1.0] * 10 + [0.0] * 10 + [1.0] * 90)
    ece, _, _ = expected_calibration_error(confidences, correct)
    assert ece < 0.05


def test_overconfidence_produces_high_ece():
    confidences = np.full(100, 0.99)
    correct = np.array([1.0] * 50 + [0.0] * 50)
    ece, _, _ = expected_calibration_error(confidences, correct)
    assert ece > 0.4


def test_confidence_of_one_lands_in_the_final_bin():
    """Left-closed bins would silently drop every perfectly-confident item."""
    ece, _, bins = expected_calibration_error(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    assert sum(b.count for b in bins) == 2


def test_auc_of_a_perfect_ranker_is_one():
    assert roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0


def test_auc_handles_ties():
    """All-equal scores carry no information; a naive pairwise count inflates
    this to 1.0."""
    assert roc_auc(np.array([0.5] * 4), np.array([0, 0, 1, 1])) == pytest.approx(0.5)


def test_brier_rewards_honest_probabilities():
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    assert brier_score(np.array([0.9, 0.9, 0.1, 0.1]), correct) < \
           brier_score(np.array([0.5, 0.5, 0.5, 0.5]), correct)


def test_threshold_sweep_coverage_is_monotone():
    report = sweep_thresholds(np.random.RandomState(0).rand(200), np.random.RandomState(1).randint(0, 2, 200))
    coverages = [row["coverage"] for row in report]
    assert coverages == sorted(coverages, reverse=True)


def test_calibration_report_renders_a_diagram():
    report = evaluate_calibration(np.array([0.2, 0.8] * 50), np.array([0.0, 1.0] * 50))
    assert "conf" in report.reliability_plot()


# -- drift -----------------------------------------------------------------

def test_psi_is_near_zero_for_identical_distributions():
    values = np.random.RandomState(0).normal(size=1000)
    assert population_stability_index(values, values) < 0.01


def test_psi_flags_a_shifted_distribution():
    rng = np.random.RandomState(0)
    assert population_stability_index(rng.normal(0, 1, 1000), rng.normal(3, 1, 1000)) > 0.25


def test_psi_survives_an_empty_bin():
    """Without smoothing, one empty bin makes the ratio infinite and a single
    missing value reports catastrophic drift."""
    rng = np.random.RandomState(0)
    value = population_stability_index(rng.normal(size=500), np.full(500, 0.5))
    assert np.isfinite(value)


def test_ks_detects_a_shift():
    rng = np.random.RandomState(0)
    assert ks_statistic(rng.normal(0, 1, 500), rng.normal(2, 1, 500)) > 0.5


def test_drift_monitor_stays_quiet_below_the_sample_floor():
    """Under ~20 samples both statistics are noise, and an alert nobody can
    trust trains people to ignore all of them."""
    monitor = DriftMonitor(features=["confidence"])
    monitor.fit_reference([{"confidence": 0.5}] * 5)
    for _ in range(5):
        monitor.observe({"confidence": 0.9})
    assert monitor.report() == []


def test_drift_monitor_alerts_on_a_real_shift():
    rng = np.random.RandomState(0)
    monitor = DriftMonitor(features=["confidence"])
    monitor.fit_reference([{"confidence": float(v)} for v in rng.normal(0.5, 0.1, 300)])
    for value in rng.normal(0.9, 0.1, 300):
        monitor.observe({"confidence": float(value)})
    assert monitor.alerts()


# -- gate ------------------------------------------------------------------

def test_gate_fails_when_a_floor_is_missed():
    from groundwork.evaluation.calibration import CalibrationReport
    from groundwork.evaluation.metrics import RetrievalScores
    from groundwork.evaluation.runner import check_gate

    retrieval = {"sparse": RetrievalScores(n=10, recall_at_5=0.10, mrr=0.10)}
    calibration = CalibrationReport(ece=0.9, mce=0.9, brier=0.5, auc=0.5,
                                    accuracy=0.5, base_rate=0.5, n=10)
    result = check_gate(retrieval, {}, calibration)
    assert not result.passed


def test_gate_flags_a_regression_against_baseline():
    from groundwork.evaluation.calibration import CalibrationReport
    from groundwork.evaluation.metrics import RetrievalScores
    from groundwork.evaluation.runner import check_gate

    retrieval = {"sparse": RetrievalScores(n=10, recall_at_5=0.86, mrr=0.76)}
    calibration = CalibrationReport(ece=0.05, mce=0.1, brier=0.2, auc=0.7,
                                    accuracy=0.7, base_rate=0.5, n=10)
    baseline = {"retrieval.recall@5": 0.95}   # we were much better yesterday
    result = check_gate(retrieval, {}, calibration, baseline=baseline)
    assert not result.passed
    assert any("baseline" in c["metric"] for c in result.checks)
