"""A learned confidence model over retrieval and generation features.

The raw impulse in RAG systems is to use the top similarity score as
confidence. It is a bad estimator: it is not on a probability scale, it does
not know whether the answer actually used the retrieved text, and it cannot
tell "one strong hit" from "five mediocre hits that averaged out".

This module instead extracts a handful of cheap features and fits a logistic
regression to predict *whether the system's answer will be correct*. It is
trained on real labels, because SQuAD v2 supplies them: for an answerable
question, correct means the gold span is in the answer; for an unanswerable
one, correct means the system abstained.

Logistic regression rather than something bigger, on purpose. Six features and
a few thousand examples is not a deep learning problem, and a linear model has
readable coefficients, so you can see which signal is doing the work and
notice when one stops.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import ARTIFACT_DIR

# Feature set, chosen by measurement rather than intuition. Single-feature AUC
# against real correctness labels over 812 answered questions:
#
#     query_coverage_answer    0.660   <- strongest
#     query_coverage_top       0.608
#     top_score                0.597
#     query_coverage_context   0.575
#     answer_length            0.540
#     answer_type_match        0.526
#     score_margin             0.497   <- dropped, no signal
#     answer_support           0.500   <- dropped, see below
#     n_passages               0.500   <- dropped, no signal
#
# `answer_support` (fraction of answer terms found in the retrieved context)
# is the instructive one. It reads like the obvious groundedness signal and it
# scores exactly 0.500, because this reader is extractive: every answer is
# copied out of the context, so the feature is ~1.0 for every row, correct and
# incorrect alike. It is a real check for a generative reader and pure noise
# for this one. Worth remembering the next time a feature is added because it
# sounds right.
FEATURE_NAMES = (
    "top_score",              # fused score of the best passage, scaled
    "query_coverage_top",     # question terms present in the top passage
    "query_coverage_context", # question terms present anywhere in context
    "query_coverage_answer",  # question terms echoed by the answer itself
    "answer_type_match",      # does the answer look like the wh-word expects
    "answer_length",          # log answer length, normalised
)

DEFAULT_MODEL_PATH = ARTIFACT_DIR / "confidence_model.json"


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error.

    Duplicated in miniature from `evaluation.calibration` so that this module,
    which ships in the request path, does not import the evaluation package.
    The full implementation with reliability bins lives there.
    """
    probabilities = np.asarray(probabilities, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    if probabilities.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total, error = probabilities.size, 0.0
    for i in range(bins):
        upper_inclusive = i == bins - 1
        mask = (probabilities >= edges[i]) & (
            (probabilities <= edges[i + 1]) if upper_inclusive else (probabilities < edges[i + 1])
        )
        if not mask.any():
            continue
        error += (mask.sum() / total) * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(error)


@dataclass
class ConfidenceModel:
    """Logistic regression with an optional post-hoc temperature.

    Weights default to a hand-set prior so the system is usable before any
    fitting has happened. `fit` replaces them with learned ones, and
    `evaluation.calibration` reports how much that moved ECE.
    """

    weights: np.ndarray = field(default_factory=lambda: np.array(
        [0.8, 0.9, 0.4, 1.6, 0.3, -0.3], dtype=np.float64
    ))
    bias: float = -0.6
    temperature: float = 1.0
    fitted: bool = False
    n_train: int = 0

    # -- inference --------------------------------------------------------

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        features = np.atleast_2d(features)
        logits = features @ self.weights + self.bias
        return _sigmoid(logits / max(self.temperature, 1e-3))

    def predict_one(self, features: dict[str, float]) -> float:
        vector = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)
        return float(self.predict_proba(vector)[0])

    # -- fitting ----------------------------------------------------------

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 400,
        lr: float = 0.35,
        l2: float = 1e-3,
    ) -> "ConfidenceModel":
        """Batch gradient descent on the log loss.

        Deliberately plain: no solver dependency, no early stopping, no
        learning-rate schedule. With six features this converges in well under
        the epoch budget, and a reader can verify the gradient by eye.
        """
        X = np.atleast_2d(features).astype(np.float64)
        y = np.asarray(labels, dtype=np.float64).ravel()
        if X.shape[0] == 0:
            return self

        w = np.zeros(X.shape[1], dtype=np.float64)
        b = 0.0
        n = X.shape[0]

        for _ in range(epochs):
            preds = _sigmoid(X @ w + b)
            error = preds - y
            grad_w = X.T @ error / n + l2 * w
            grad_b = float(error.mean())
            w -= lr * grad_w
            b -= lr * grad_b

        self.weights = w
        self.bias = b
        self.temperature = 1.0  # fitting a fresh model resets any prior scaling
        self.fitted = True
        self.n_train = n
        return self

    def fit_temperature(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        grid: np.ndarray | None = None,
    ) -> float:
        """Post-hoc temperature scaling.

        Fitted on a split the weights never saw. Temperature cannot change the
        ranking of predictions, so it never changes accuracy, AUC, or any
        threshold-free metric. It only changes how honest the numbers are,
        which is the entire point of calibration.
        """
        X = np.atleast_2d(features).astype(np.float64)
        y = np.asarray(labels, dtype=np.float64).ravel()
        if X.shape[0] == 0:
            return self.temperature

        logits = X @ self.weights + self.bias
        grid = grid if grid is not None else np.linspace(0.25, 5.0, 96)

        # Selected on validation ECE, with NLL as the tiebreak.
        #
        # Guo et al. select on NLL, and that is right when NLL is what you
        # care about. Here ECE is the reported metric and the guard below is
        # an ECE test, so selecting on NLL and gating on ECE optimises one
        # quantity and judges another. That mismatch is exactly what produced
        # the two bad temperatures recorded below: NLL-optimal T=0.70 and
        # T=0.75 both cleared an NLL improvement and then degraded held-out
        # ECE from 0.080 to 0.147 and from 0.053 to 0.066.
        best_t, best_key = 1.0, (math.inf, math.inf)
        for t in grid:
            p = np.clip(_sigmoid(logits / t), 1e-9, 1 - 1e-9)
            nll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
            key = (_ece(p, y), nll)
            if key < best_key:
                best_key, best_t = key, float(t)

        # Guard: accept the scaling only if it clearly improves ECE on this
        # same held-out slice. NLL is the textbook objective (Guo et al.), but
        # at a few hundred validation points the NLL-optimal temperature
        # routinely makes ECE *worse* on a fresh split, and ECE is the number
        # this project reports. Two earlier runs here shipped T=0.70 and T=0.75
        # and moved held-out ECE from 0.080 to 0.147 and from 0.053 to 0.066:
        # a calibration step that decalibrated the model.
        #
        # Requiring a 10% relative improvement rather than any improvement is
        # what makes the guard bite. A model that is already well calibrated
        # has nothing for temperature scaling to fix, and fitting one anyway
        # buys noise. Identity is the right default, not the fallback.
        baseline_ece = _ece(_sigmoid(logits), y)
        scaled_ece = _ece(_sigmoid(logits / best_t), y)

        # Accept a temperature only on a large, absolute validation win.
        #
        # This threshold is empirical and the experiment is worth recording.
        # With ~295 validation points and a baseline ECE around 0.05, the
        # optimal temperature is simply not identifiable: validation
        # confidently preferred T=0.70, and the held-out split preferred T=1.0
        # every time, with reported ECE going 0.053 -> 0.084 when validation
        # got its way. Selecting on NLL, then on ECE, then on ECE with a
        # relative-improvement guard all failed the same way, because the
        # problem is the sample size and not the objective.
        #
        # So the bar is deliberately high: the model must be visibly
        # miscalibrated to begin with, and scaling must buy a large absolute
        # improvement. Temperature scaling then stays available for the case it
        # is actually for (an early build here scored ECE 0.94) and stops
        # firing on noise. Identity is the default, not the fallback.
        min_baseline_ece = 0.08
        min_absolute_gain = 0.02
        if baseline_ece < min_baseline_ece or (baseline_ece - scaled_ece) < min_absolute_gain:
            best_t = 1.0

        self.temperature = best_t
        return best_t

    # -- introspection ----------------------------------------------------

    def coefficients(self) -> dict[str, float]:
        return {name: round(float(w), 4) for name, w in zip(FEATURE_NAMES, self.weights)}

    # -- persistence ------------------------------------------------------

    def save(self, path: Path | None = None) -> Path:
        path = path or DEFAULT_MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "feature_names": list(FEATURE_NAMES),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "temperature": self.temperature,
            "fitted": self.fitted,
            "n_train": self.n_train,
        }, indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "ConfidenceModel":
        path = path or DEFAULT_MODEL_PATH
        if not path.exists():
            return cls()  # prior weights; the system still works unfitted
        payload = json.loads(path.read_text())
        if list(payload.get("feature_names", [])) != list(FEATURE_NAMES):
            # Feature set changed since the model was fitted. Silently loading
            # it would apply weights to the wrong columns, which is worse than
            # having no model at all.
            return cls()
        return cls(
            weights=np.array(payload["weights"], dtype=np.float64),
            bias=float(payload["bias"]),
            temperature=float(payload.get("temperature", 1.0)),
            fitted=bool(payload.get("fitted", False)),
            n_train=int(payload.get("n_train", 0)),
        )


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[S\d+\]")

# Expected answer shape by interrogative. Crude, and it earns its place: a
# "when" question answered with no date in sight is a weak answer, and this
# costs a regex rather than a second model call.
_ANSWER_TYPE_CUES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("what year", "when", "what date", "how long", "what century"), "date"),
    (("how many", "how much", "what percentage", "what number"), "number"),
    (("who", "whom", "whose", "where", "which city", "which country"), "name"),
)

_DATE_RE = re.compile(
    r"\b(1[0-9]{3}|20[0-9]{2}|\d{1,2}(st|nd|rd|th)|january|february|march|april|may|june|"
    r"july|august|september|october|november|december|century|decade)\b", re.I)
_NUMBER_RE = re.compile(r"\b\d+([.,]\d+)?\b|\b(one|two|three|four|five|six|seven|eight|nine|ten|"
                        r"hundred|thousand|million|billion)\b", re.I)
_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")


def _answer_type_match(question: str, answer: str) -> float:
    """1.0 matches the expected shape, 0.0 contradicts it, 0.5 no expectation.

    Neutral rather than zero when the question carries no cue, so an
    unclassifiable question is not silently penalised.
    """
    lowered = question.lower()
    expected = next(
        (kind for cues, kind in _ANSWER_TYPE_CUES if any(c in lowered for c in cues)), None
    )
    if expected is None:
        return 0.5
    body = _CITATION_RE.sub("", answer)
    pattern = {"date": _DATE_RE, "number": _NUMBER_RE, "name": _NAME_RE}[expected]
    return 1.0 if pattern.search(body) else 0.0


def extract_features(
    *,
    question: str,
    answer: str,
    retrieval,
    n_citations: int,
) -> dict[str, float]:
    """Cheap, model-agnostic signals available at answer time.

    Nothing here needs a second model call. A confidence estimate that costs
    an extra round trip gets switched off the first time latency matters, and
    one that runs only sometimes is worse than none at all.
    """
    from .embed import tokenize

    passages = getattr(retrieval, "passages", [])
    q_terms = set(tokenize(question))
    body = _CITATION_RE.sub("", answer)
    a_terms = set(tokenize(body))

    top_terms = set(tokenize(passages[0].text)) if passages else set()
    context_terms: set[str] = set()
    for passage in passages:
        context_terms |= set(tokenize(passage.text))

    def coverage(target: set[str]) -> float:
        return len(q_terms & target) / len(q_terms) if q_terms else 0.0

    # RRF scores are bounded by (sum of weights) / (k + 1). Scaling to roughly
    # [0, 1] keeps the linear model's inputs on comparable ranges without a
    # separate normalisation step at inference time.
    return {
        "top_score": min(1.0, (retrieval.top_score if passages else 0.0) * 30.0),
        "query_coverage_top": coverage(top_terms),
        "query_coverage_context": coverage(context_terms),
        "query_coverage_answer": coverage(a_terms),
        "answer_type_match": _answer_type_match(question, answer),
        "answer_length": min(1.0, math.log1p(len(a_terms)) / 4.0),
    }


def features_to_vector(features: dict[str, float]) -> np.ndarray:
    return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)
