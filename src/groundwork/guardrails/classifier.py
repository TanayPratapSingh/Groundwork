"""A small text classifier: hashed n-grams plus logistic regression.

Used for the injection and jailbreak rails. Deliberately not a transformer.

The reasoning is a latency budget, not nostalgia. An input rail runs on the
critical path of every request, before the model call, and it competes with
the model call for the user's patience. A rail that adds 300ms of GPU
inference to save 900ms of a refused generation is a bad trade at the p50 and
a worse one at the p99. This classifier scores a prompt in microseconds on a
CPU, which means it can be always-on.

The cost of that choice is real and is measured rather than hidden: see the
gap between test-split and held-out-attack-set recall in docs/06-results.md.
Word n-grams generalise to paraphrase poorly, so a lexically novel attack gets
past it. That is exactly why the rails are layered and why the output rail
verifies grounding independently of whether the input rail fired.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import ARTIFACT_DIR

_WORD_RE = re.compile(r"[a-z0-9']+")


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def featurize(text: str, dim: int) -> np.ndarray:
    """Hash word unigrams, bigrams, and character 4-grams into one vector.

    Character n-grams are what catch the obfuscations that word features miss:
    "ig nore previous", "1gnore", "I-G-N-O-R-E". Word n-grams carry the phrase
    structure. Together they cover more attack surface than either alone.
    """
    vec = np.zeros(dim, dtype=np.float32)
    lowered = text.lower()
    words = _WORD_RE.findall(lowered)

    features: list[str] = [f"w:{w}" for w in words]
    features += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]

    compact = re.sub(r"\s+", " ", lowered)
    features += [f"c:{compact[i:i + 4]}" for i in range(0, max(0, len(compact) - 3), 2)]

    # Three cheap structural signals. Injections skew long, punctuation-heavy,
    # and imperative in a way the bag of n-grams does not capture directly.
    if words:
        features.append(f"len:{min(9, len(words) // 25)}")
    if compact.count("\n") > 3:
        features.append("multiline")
    if sum(1 for c in text if c.isupper()) > len(text) * 0.3 and len(text) > 20:
        features.append("shouting")

    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, "big")
        vec[raw % dim] += 1.0 if (raw >> 63) & 1 else -1.0

    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


@dataclass
class HashedTextClassifier:
    """Binary logistic regression over hashed text features."""

    dim: int = 2048
    weights: np.ndarray | None = None
    bias: float = 0.0
    threshold: float = 0.5
    label: str = "positive"
    n_train: int = 0

    # -- training ---------------------------------------------------------

    def fit(
        self,
        texts: list[str],
        labels: list[int],
        epochs: int = 300,
        lr: float = 0.5,
        l2: float = 2e-4,
        class_balance: bool = True,
    ) -> "HashedTextClassifier":
        X = np.vstack([featurize(t, self.dim) for t in texts]).astype(np.float64)
        y = np.asarray(labels, dtype=np.float64)
        n = len(y)
        if n == 0:
            return self

        # Attack corpora are usually imbalanced. Without reweighting, gradient
        # descent finds the "predict benign" solution, which scores well on
        # accuracy and has zero recall, which is the metric that matters.
        if class_balance and 0 < y.sum() < n:
            w_pos = n / (2.0 * y.sum())
            w_neg = n / (2.0 * (n - y.sum()))
            sample_w = np.where(y == 1, w_pos, w_neg)
        else:
            sample_w = np.ones(n)
        sample_w /= sample_w.mean()

        w = np.zeros(self.dim, dtype=np.float64)
        b = 0.0
        for _ in range(epochs):
            preds = _sigmoid(X @ w + b)
            error = (preds - y) * sample_w
            w -= lr * (X.T @ error / n + l2 * w)
            b -= lr * float(error.mean())

        self.weights = w
        self.bias = b
        self.n_train = n
        return self

    def tune_threshold(
        self,
        texts: list[str],
        labels: list[int],
        max_fpr: float = 0.10,
    ) -> float:
        """Pick the threshold maximising recall subject to a false-positive cap.

        The cap is the product decision, and it belongs here rather than in a
        notebook: a rail with 99% recall and a 30% false-positive rate blocks
        a third of legitimate traffic and will be disabled within a week. The
        cap is what makes the rail survivable in production.
        """
        if self.weights is None or not texts:
            return self.threshold

        scores = self.score_many(texts)
        y = np.asarray(labels)
        best_threshold, best_recall = 1.0, -1.0

        for candidate in np.unique(np.round(scores, 4)):
            predicted = scores >= candidate
            tp = int((predicted & (y == 1)).sum())
            fp = int((predicted & (y == 0)).sum())
            fn = int((~predicted & (y == 1)).sum())
            tn = int((~predicted & (y == 0)).sum())
            recall = tp / (tp + fn) if tp + fn else 0.0
            fpr = fp / (fp + tn) if fp + tn else 0.0
            if fpr <= max_fpr and recall > best_recall:
                best_recall, best_threshold = recall, float(candidate)

        self.threshold = best_threshold
        return best_threshold

    # -- inference --------------------------------------------------------

    def score(self, text: str) -> float:
        if self.weights is None:
            return 0.0
        return float(_sigmoid(featurize(text, self.dim).astype(np.float64) @ self.weights + self.bias))

    def score_many(self, texts: list[str]) -> np.ndarray:
        if self.weights is None or not texts:
            return np.zeros(len(texts))
        X = np.vstack([featurize(t, self.dim) for t in texts]).astype(np.float64)
        return _sigmoid(X @ self.weights + self.bias)

    def predict(self, text: str) -> bool:
        return self.score(text) >= self.threshold

    @property
    def trained(self) -> bool:
        return self.weights is not None

    # -- persistence ------------------------------------------------------

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            weights=self.weights if self.weights is not None else np.zeros(0),
            meta=np.array([json.dumps({
                "dim": self.dim, "bias": self.bias, "threshold": self.threshold,
                "label": self.label, "n_train": self.n_train,
            })]),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "HashedTextClassifier | None":
        if not path.exists():
            return None
        payload = np.load(path, allow_pickle=False)
        meta = json.loads(str(payload["meta"][0]))
        weights = payload["weights"]
        model = cls(
            dim=meta["dim"], bias=meta["bias"], threshold=meta["threshold"],
            label=meta["label"], n_train=meta["n_train"],
        )
        model.weights = weights if weights.size else None
        return model


def model_path(name: str) -> Path:
    return ARTIFACT_DIR / f"detector_{name}.npz"
