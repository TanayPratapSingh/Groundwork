"""Confidence calibration.

A calibrated system is one where "0.8 confident" means it is right about 80
percent of the time. That is a stronger and much more useful property than
accuracy, because it is what makes a threshold meaningful. Without it, the
number in `confidence: 0.83` is an ordering with a decimal point on it, and
any policy built on a cutoff is guessing.

The metric is **expected calibration error**: bin predictions by confidence,
and average the gap between confidence and observed accuracy in each bin,
weighted by bin population.

    ECE = sum over bins of (n_b / N) * |accuracy_b - confidence_b|

Two properties of ECE worth stating plainly, since it gets quoted as though it
were a score:

- **It is not accuracy.** A model that outputs 0.5 for everything and is right
  half the time has near-zero ECE and no discriminative value at all. ECE must
  always be read next to AUC or accuracy, never alone.
- **It depends on the binning.** Fifteen bins is the common default and is
  what this uses; comparing an ECE across projects with different bin counts
  is not a comparison.

The fix, when ECE is high, is temperature scaling: divide the logits by a
single scalar fitted on held-out data. It cannot change any ranking, therefore
cannot change accuracy or AUC, and only makes the probabilities honest. One
parameter, so it does not meaningfully overfit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Bin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        return abs(self.accuracy - self.mean_confidence)

    def as_dict(self) -> dict:
        return {
            "range": [round(self.lower, 3), round(self.upper, 3)],
            "count": self.count,
            "mean_confidence": round(self.mean_confidence, 4),
            "accuracy": round(self.accuracy, 4),
            "gap": round(self.gap, 4),
        }


@dataclass
class CalibrationReport:
    ece: float
    mce: float
    brier: float
    auc: float
    accuracy: float
    base_rate: float
    n: int
    bins: list[Bin] = field(default_factory=list)
    temperature: float = 1.0

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "ece": round(self.ece, 4),
            "mce": round(self.mce, 4),
            "brier": round(self.brier, 4),
            "auc": round(self.auc, 4),
            "accuracy": round(self.accuracy, 4),
            "base_rate": round(self.base_rate, 4),
            "temperature": round(self.temperature, 4),
            "bins": [b.as_dict() for b in self.bins],
        }

    def reliability_plot(self, width: int = 42) -> str:
        """ASCII reliability diagram.

        In a terminal, next to the numbers, where someone will actually look at
        it. A calibration plot that lives in a notebook nobody opens has never
        prevented a bad release.
        """
        lines = ["  conf | accuracy                                  |  n",
                 "  -----+-------------------------------------------+------"]
        for b in self.bins:
            if b.count == 0:
                continue
            filled = int(round(b.accuracy * width))
            marker = int(round(b.mean_confidence * width))
            row = ["·"] * width
            for i in range(filled):
                row[i] = "█"
            if 0 <= marker < width:
                row[marker] = "│"  # where the model claimed it would land
            lines.append(f"  {b.mean_confidence:4.2f} | {''.join(row)} | {b.count:5d}")
        lines.append("  (█ observed accuracy, │ stated confidence: aligned means calibrated)")
        return "\n".join(lines)


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> tuple[float, float, list[Bin]]:
    confidences = np.asarray(confidences, dtype=float).ravel()
    correct = np.asarray(correct, dtype=float).ravel()
    if confidences.size == 0:
        return 0.0, 0.0, []

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = confidences.size
    ece, mce, bins = 0.0, 0.0, []

    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        # Left-closed, right-open, except the last bin which must include 1.0
        # or every perfectly-confident prediction silently vanishes.
        mask = (confidences >= low) & (confidences < high) if i < n_bins - 1 else \
               (confidences >= low) & (confidences <= high)
        count = int(mask.sum())
        if count == 0:
            bins.append(Bin(low, high, 0, 0.0, 0.0))
            continue
        mean_conf = float(confidences[mask].mean())
        accuracy = float(correct[mask].mean())
        gap = abs(accuracy - mean_conf)
        ece += (count / total) * gap
        mce = max(mce, gap)
        bins.append(Bin(low, high, count, mean_conf, accuracy))

    return float(ece), float(mce), bins


def brier_score(confidences: np.ndarray, correct: np.ndarray) -> float:
    """Mean squared error of the probabilities.

    Decomposes into calibration plus refinement, so unlike ECE it is sensitive
    to both honesty and discrimination. A useful single number when you can
    only report one.
    """
    confidences = np.asarray(confidences, dtype=float).ravel()
    correct = np.asarray(correct, dtype=float).ravel()
    if confidences.size == 0:
        return 0.0
    return float(np.mean((confidences - correct) ** 2))


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC via the rank-sum identity, with correct handling of ties.

    Ties matter here: a detector that assigns the same score to many inputs
    gets a materially inflated AUC under a naive pairwise count.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.size, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1, dtype=float)

    # Average ranks within tied groups.
    sorted_scores = scores[order]
    start = 0
    for i in range(1, sorted_scores.size + 1):
        if i == sorted_scores.size or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = (start + 1 + i) / 2.0
            start = i

    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def evaluate_calibration(
    confidences, correct, n_bins: int = 15, temperature: float = 1.0
) -> CalibrationReport:
    confidences = np.asarray(confidences, dtype=float).ravel()
    correct = np.asarray(correct, dtype=float).ravel()
    ece, mce, bins = expected_calibration_error(confidences, correct, n_bins)
    return CalibrationReport(
        ece=ece,
        mce=mce,
        brier=brier_score(confidences, correct),
        auc=roc_auc(confidences, correct),
        accuracy=float(((confidences >= 0.5) == (correct == 1)).mean()) if confidences.size else 0.0,
        base_rate=float(correct.mean()) if correct.size else 0.0,
        n=int(confidences.size),
        bins=bins,
        temperature=temperature,
    )


def sweep_thresholds(
    confidences, correct, steps: int = 41
) -> list[dict]:
    """Precision, recall, and coverage across the answer/abstain cutoff.

    This is the table the product decision is actually made from. "Answer only
    above 0.6" is not a technical choice, it is a statement about what fraction
    of questions the system should decline, and the only way to make it
    deliberately is to see the whole curve.
    """
    confidences = np.asarray(confidences, dtype=float).ravel()
    correct = np.asarray(correct, dtype=float).ravel()
    rows = []
    for threshold in np.linspace(0.0, 1.0, steps):
        answered = confidences >= threshold
        n_answered = int(answered.sum())
        coverage = n_answered / confidences.size if confidences.size else 0.0
        precision = float(correct[answered].mean()) if n_answered else 1.0
        # Recall over the whole set: of everything we could have got right,
        # how much did we actually deliver rather than decline.
        recall = float(correct[answered].sum() / correct.sum()) if correct.sum() else 0.0
        rows.append({
            "threshold": round(float(threshold), 3),
            "coverage": round(coverage, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
            "n_answered": n_answered,
        })
    return rows
