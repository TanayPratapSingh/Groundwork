"""Distribution drift monitoring.

A RAG system degrades silently. Nothing errors, latency is flat, and the
answers get worse because the questions arriving are no longer the questions
the thresholds were fitted on. Accuracy monitoring cannot catch this in
production because production has no labels; distribution monitoring can,
because it needs only the inputs.

Two complementary statistics:

**PSI** (population stability index) bins a reference window and a current
window and sums the weighted log ratio. It is the standard in credit risk for
good reason: it is stable at small sample sizes and its thresholds are
industry-conventional rather than invented per project.

    < 0.10   no meaningful shift
    < 0.25   moderate shift, worth investigating
    >= 0.25  significant shift, treat the fitted thresholds as stale

**KS statistic** is the largest gap between the two empirical CDFs. It is
binning-free, so it catches shifts that happen to fall inside a PSI bin, and
it disagrees with PSI often enough to be worth having.

What to monitor: retrieval top score, score margin, confidence, groundedness,
and abstention rate. Confidence drifting up while groundedness drifts down is
the specific signature of a system that has started bluffing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DriftResult:
    feature: str
    psi: float
    ks: float
    reference_mean: float
    current_mean: float
    n_reference: int
    n_current: int

    @property
    def severity(self) -> str:
        if self.psi >= 0.25:
            return "significant"
        if self.psi >= 0.10:
            return "moderate"
        return "stable"

    @property
    def alert(self) -> bool:
        return self.psi >= 0.25 or self.ks >= 0.20

    def as_dict(self) -> dict:
        return {
            "feature": self.feature,
            "psi": round(self.psi, 4),
            "ks": round(self.ks, 4),
            "severity": self.severity,
            "alert": self.alert,
            "reference_mean": round(self.reference_mean, 4),
            "current_mean": round(self.current_mean, 4),
            "n_reference": self.n_reference,
            "n_current": self.n_current,
        }


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    reference = np.asarray(reference, dtype=float).ravel()
    current = np.asarray(current, dtype=float).ravel()
    if reference.size == 0 or current.size == 0:
        return 0.0

    # Quantile edges from the reference window, so bins hold roughly equal mass
    # under the baseline. Fixed-width bins on a skewed feature put almost
    # everything in one bucket and report zero drift no matter what happens.
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    # Laplace smoothing. Without it an empty bin makes the ratio infinite and
    # a single missing value reports catastrophic drift.
    ref_pct = (ref_counts + 0.5) / (ref_counts.sum() + 0.5 * len(ref_counts))
    cur_pct = (cur_counts + 0.5) / (cur_counts.sum() + 0.5 * len(cur_counts))

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    reference = np.sort(np.asarray(reference, dtype=float).ravel())
    current = np.sort(np.asarray(current, dtype=float).ravel())
    if reference.size == 0 or current.size == 0:
        return 0.0

    grid = np.union1d(reference, current)
    cdf_ref = np.searchsorted(reference, grid, side="right") / reference.size
    cdf_cur = np.searchsorted(current, grid, side="right") / current.size
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


class DriftMonitor:
    """Compares a fixed reference window against a rolling current window."""

    def __init__(self, features: list[str] | None = None, bins: int = 10) -> None:
        self.features = features or [
            "top_score", "score_margin", "query_coverage",
            "answer_support", "confidence", "groundedness",
        ]
        self.bins = bins
        self._reference: dict[str, list[float]] = {f: [] for f in self.features}
        self._current: dict[str, list[float]] = {f: [] for f in self.features}

    def fit_reference(self, observations: list[dict]) -> "DriftMonitor":
        self._reference = {f: [] for f in self.features}
        for obs in observations:
            for feature in self.features:
                if feature in obs and obs[feature] is not None:
                    self._reference[feature].append(float(obs[feature]))
        return self

    def observe(self, observation: dict) -> None:
        for feature in self.features:
            if feature in observation and observation[feature] is not None:
                self._current[feature].append(float(observation[feature]))

    def reset_current(self) -> None:
        self._current = {f: [] for f in self.features}

    def report(self) -> list[DriftResult]:
        results = []
        for feature in self.features:
            ref = np.array(self._reference.get(feature, []), dtype=float)
            cur = np.array(self._current.get(feature, []), dtype=float)
            if ref.size < 20 or cur.size < 20:
                # Under ~20 samples both statistics are noise. Reporting a
                # number anyway trains people to ignore the alert.
                continue
            results.append(DriftResult(
                feature=feature,
                psi=population_stability_index(ref, cur, self.bins),
                ks=ks_statistic(ref, cur),
                reference_mean=float(ref.mean()),
                current_mean=float(cur.mean()),
                n_reference=int(ref.size),
                n_current=int(cur.size),
            ))
        return results

    def alerts(self) -> list[DriftResult]:
        return [r for r in self.report() if r.alert]
