"""Loaders for the labelled attack and privacy corpora.

Each loader returns a `LabelledSet` with the split preserved, because the
split is doing real work: detectors are fitted on `train` and reported on
`test`, and the Gandalf corpus is never fitted on at all.

Gandalf deserves a note. It is 1,000 injection attempts written by players
trying to beat a live guarded system, and it has no benign class, so it can
only measure recall. That is precisely what makes it the most informative set
here: it is unseen, adversarial, and produced by people with an incentive to
be novel rather than by an annotation pipeline. The gap between test-split
recall and Gandalf recall is the honest estimate of how much of a detector's
score is memorisation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config import RAW_DIR


class DatasetMissing(FileNotFoundError):
    def __init__(self, path: Path) -> None:
        super().__init__(
            f"{path} not found. Fetch the real datasets first:\n\n"
            f"    python3 scripts/fetch_data.py\n"
        )


@dataclass
class LabelledSet:
    """Texts with binary labels, plus the split each item came from."""

    name: str
    texts: list[str] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    splits: list[str] = field(default_factory=list)
    positive_meaning: str = "attack"

    def __len__(self) -> int:
        return len(self.texts)

    def split(self, name: str) -> "LabelledSet":
        keep = [i for i, s in enumerate(self.splits) if s == name]
        return LabelledSet(
            name=f"{self.name}[{name}]",
            texts=[self.texts[i] for i in keep],
            labels=[self.labels[i] for i in keep],
            splits=[self.splits[i] for i in keep],
            positive_meaning=self.positive_meaning,
        )

    @property
    def n_positive(self) -> int:
        return sum(self.labels)

    @property
    def n_negative(self) -> int:
        return len(self.labels) - self.n_positive

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n": len(self),
            "positive": self.n_positive,
            "negative": self.n_negative,
            "splits": sorted(set(self.splits)),
        }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise DatasetMissing(path)
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_prompt_injections(path: Path | None = None) -> LabelledSet:
    """deepset/prompt-injections. label 1 = injection."""
    rows = _read_jsonl(path or RAW_DIR / "prompt_injections.jsonl")
    return LabelledSet(
        name="deepset/prompt-injections",
        texts=[r["text"] for r in rows],
        labels=[int(r["label"]) for r in rows],
        splits=[r.get("_split", "train") for r in rows],
        positive_meaning="prompt injection",
    )


def load_jailbreaks(path: Path | None = None) -> LabelledSet:
    """jackhhao/jailbreak-classification. type in {benign, jailbreak}."""
    rows = _read_jsonl(path or RAW_DIR / "jailbreaks.jsonl")
    return LabelledSet(
        name="jackhhao/jailbreak-classification",
        texts=[r["prompt"] for r in rows],
        labels=[1 if str(r.get("type", "")).lower() == "jailbreak" else 0 for r in rows],
        splits=[r.get("_split", "train") for r in rows],
        positive_meaning="jailbreak",
    )


def load_gandalf(path: Path | None = None) -> LabelledSet:
    """Lakera/gandalf_ignore_instructions. Attacks only, so recall only."""
    rows = _read_jsonl(path or RAW_DIR / "gandalf_injections.jsonl")
    return LabelledSet(
        name="Lakera/gandalf_ignore_instructions",
        texts=[r["text"] for r in rows],
        labels=[1] * len(rows),
        splits=[r.get("_split", "train") for r in rows],
        positive_meaning="prompt injection (held out)",
    )


# --------------------------------------------------------------------------
# PII
# --------------------------------------------------------------------------

# ai4privacy annotates 56 label types, many of which are not PII in any
# regulatory sense (JOBAREA, CURRENCYSYMBOL, TIME). Mapping only the ones this
# project's detectors actually claim keeps the recall figure meaningful:
# scoring against all 56 would report a low number that says nothing about the
# detectors and everything about the mismatch in scope.
PII_LABEL_MAP: dict[str, str] = {
    "EMAIL": "EMAIL",
    "PHONENUMBER": "PHONE",
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "SSN": "SSN",
    "IBAN": "IBAN",
    "IPV4": "IP_ADDRESS",
    "IPV6": "IP_ADDRESS",
    "IP": "IP_ADDRESS",
    "DOB": "DATE_OF_BIRTH",
}


@dataclass
class PiiExample:
    text: str
    spans: list[dict]  # {value, start, end, label} with label already mapped

    @property
    def entity_types(self) -> set[str]:
        return {s["label"] for s in self.spans}


def load_pii(path: Path | None = None, english_only: bool = True) -> list[PiiExample]:
    """ai4privacy/pii-masking-200k, restricted to the mapped entity types."""
    rows = _read_jsonl(path or RAW_DIR / "pii.jsonl")
    examples: list[PiiExample] = []

    for row in rows:
        if english_only and row.get("language") not in (None, "en", "English"):
            continue
        mask = row.get("privacy_mask") or []
        if isinstance(mask, str):
            try:
                mask = json.loads(mask.replace("'", '"'))
            except json.JSONDecodeError:
                continue

        spans = [
            {
                "value": m.get("value", ""),
                "start": int(m.get("start", 0)),
                "end": int(m.get("end", 0)),
                "label": PII_LABEL_MAP[m["label"]],
            }
            for m in mask
            if isinstance(m, dict) and m.get("label") in PII_LABEL_MAP
        ]
        if spans:
            examples.append(PiiExample(text=row["source_text"], spans=spans))

    return examples


def available() -> dict[str, bool]:
    """Which corpora are present, for graceful degradation in the runner."""
    return {
        "prompt_injections": (RAW_DIR / "prompt_injections.jsonl").exists(),
        "jailbreaks": (RAW_DIR / "jailbreaks.jsonl").exists(),
        "gandalf": (RAW_DIR / "gandalf_injections.jsonl").exists(),
        "pii": (RAW_DIR / "pii.jsonl").exists(),
        "squad": (RAW_DIR / "squad_v2_dev.json").exists(),
    }
