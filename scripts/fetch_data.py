#!/usr/bin/env python3
"""Download every dataset this project is measured on.

No data is generated, paraphrased, or hand-authored anywhere in this repo.
Each source below is a public, independently published dataset with its own
citation and licence, recorded in data/manifest.json alongside a SHA-256 of
exactly what was downloaded. Re-running this script is how a reader reproduces
the numbers in docs/06-results.md.

    python3 scripts/fetch_data.py            # everything
    python3 scripts/fetch_data.py --only squad
    python3 scripts/fetch_data.py --force    # ignore existing files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw"
MANIFEST = REPO_ROOT / "data" / "manifest.json"

HF_ROWS = "https://datasets-server.huggingface.co/rows"
USER_AGENT = "groundwork/0.1 (+https://github.com/TanayPratapSingh/groundwork)"

# --------------------------------------------------------------------------
# Source registry
# --------------------------------------------------------------------------
# `role` explains what the dataset is load-bearing for, so nobody has to guess
# why a dependency exists.

DIRECT_SOURCES = {
    "squad_v2_dev": {
        "url": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
        "filename": "squad_v2_dev.json",
        "licence": "CC BY-SA 4.0",
        "citation": (
            "Rajpurkar, Jia & Liang (2018). Know What You Don't Know: "
            "Unanswerable Questions for SQuAD. ACL 2018."
        ),
        "homepage": "https://rajpurkar.github.io/SQuAD-explorer/",
        "role": (
            "Retrieval corpus (Wikipedia paragraphs), question set with gold "
            "answer spans, and the unanswerable labels that give the abstain "
            "policy and the calibration study real ground truth."
        ),
    },
}

HF_SOURCES = {
    "prompt_injections": {
        "dataset": "deepset/prompt-injections",
        "config": "default",
        "splits": ["train", "test"],
        "filename": "prompt_injections.jsonl",
        "licence": "CC BY 4.0",
        "citation": "deepset. prompt-injections (Hugging Face Datasets).",
        "homepage": "https://huggingface.co/datasets/deepset/prompt-injections",
        "role": "Labelled benign/injection prompts: precision and recall for the injection rail.",
    },
    "jailbreaks": {
        "dataset": "jackhhao/jailbreak-classification",
        "config": "default",
        "splits": ["train", "test"],
        "filename": "jailbreaks.jsonl",
        "licence": "See dataset card",
        "citation": "jackhhao. jailbreak-classification (Hugging Face Datasets).",
        "homepage": "https://huggingface.co/datasets/jackhhao/jailbreak-classification",
        "role": "Labelled jailbreak/benign prompts: precision and recall for the jailbreak rail.",
    },
    "gandalf_injections": {
        "dataset": "Lakera/gandalf_ignore_instructions",
        "config": "default",
        "splits": ["train", "validation", "test"],
        "filename": "gandalf_injections.jsonl",
        "licence": "MIT",
        "citation": "Lakera AI. gandalf_ignore_instructions (Hugging Face Datasets).",
        "homepage": "https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions",
        "role": (
            "Real adversarial injection attempts written by players trying to "
            "beat a live system. Held out as an unseen attack set, so the "
            "injection rail is never tuned on it."
        ),
    },
    "pii": {
        "dataset": "ai4privacy/pii-masking-200k",
        "config": "default",
        "splits": ["train"],
        "filename": "pii.jsonl",
        "licence": "See dataset card (research use)",
        "citation": "ai4privacy. pii-masking-200k (Hugging Face Datasets).",
        "homepage": "https://huggingface.co/datasets/ai4privacy/pii-masking-200k",
        "role": "Texts with annotated PII spans: recall of the redaction rail per entity type.",
        # 209k rows is far more than the rail needs and the API paginates at
        # 100. A fixed prefix keeps the download honest and reproducible.
        "max_rows": 1000,
    },
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _get(url: str, retries: int = 6, timeout: int = 90) -> bytes:
    """GET with backoff.

    The datasets-server rate-limits anonymous callers hard once you have pulled
    a few thousand rows, and it answers with a 429 HTML page rather than JSON.
    Backoff on 429 starts at 5s rather than 1s because the shorter schedule
    reliably exhausts its retries without ever clearing the limit.
    """
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            if exc.code in (500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            raise
        except urllib.error.URLError as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {retries} attempts: {url}") from last


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


# --------------------------------------------------------------------------
# Fetchers
# --------------------------------------------------------------------------

def fetch_direct(key: str, spec: dict, force: bool) -> dict:
    dest = RAW / spec["filename"]
    if dest.exists() and not force:
        print(f"  [skip] {spec['filename']} already present")
    else:
        print(f"  [get ] {spec['url']}")
        dest.write_bytes(_get(spec["url"]))

    payload = json.loads(dest.read_text())
    paragraphs = sum(len(a["paragraphs"]) for a in payload["data"])
    questions = [q for a in payload["data"] for p in a["paragraphs"] for q in p["qas"]]
    unanswerable = sum(1 for q in questions if q.get("is_impossible"))

    return {
        "source": spec["url"],
        "file": spec["filename"],
        "sha256": _sha256(dest),
        "bytes": dest.stat().st_size,
        "licence": spec["licence"],
        "citation": spec["citation"],
        "homepage": spec["homepage"],
        "role": spec["role"],
        "stats": {
            "articles": len(payload["data"]),
            "paragraphs": paragraphs,
            "questions": len(questions),
            "answerable": len(questions) - unanswerable,
            "unanswerable": unanswerable,
        },
    }


def fetch_hf(key: str, spec: dict, force: bool) -> dict:
    dest = RAW / spec["filename"]
    if dest.exists() and not force:
        print(f"  [skip] {spec['filename']} already present")
        rows = sum(1 for _ in dest.open())
    else:
        collected: list[dict] = []
        cap = spec.get("max_rows")
        for split in spec["splits"]:
            offset, split_rows = 0, 0
            while True:
                if cap is not None and len(collected) >= cap:
                    break
                params = urllib.parse.urlencode({
                    "dataset": spec["dataset"],
                    "config": spec["config"],
                    "split": split,
                    "offset": offset,
                    "length": 100,
                })
                payload = json.loads(_get(f"{HF_ROWS}?{params}"))
                batch = payload.get("rows", [])
                if not batch:
                    break
                for item in batch:
                    if cap is not None and len(collected) >= cap:
                        break
                    record = dict(item["row"])
                    record["_split"] = split
                    collected.append(record)
                split_rows += len(batch)
                offset += len(batch)
                total = payload.get("num_rows_total")
                if total is not None and offset >= total:
                    break
                time.sleep(0.35)  # be a good citizen on a free public API
            print(f"  [get ] {spec['dataset']}:{split} -> {split_rows} rows")

        with dest.open("w") as fh:
            for record in collected:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        rows = len(collected)

    return {
        "source": f"huggingface:{spec['dataset']}",
        "file": spec["filename"],
        "sha256": _sha256(dest),
        "bytes": dest.stat().st_size,
        "licence": spec["licence"],
        "citation": spec["citation"],
        "homepage": spec["homepage"],
        "role": spec["role"],
        "stats": {"rows": rows, "splits": spec["splits"]},
    }


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="fetch a subset by key")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only) if args.only else None

    manifest = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text()).get("datasets", {})

    print("Fetching real datasets. Nothing in this project is synthesised.\n")

    for key, spec in DIRECT_SOURCES.items():
        if wanted and key not in wanted:
            continue
        print(f"{key}:")
        manifest[key] = fetch_direct(key, spec, args.force)

    for key, spec in HF_SOURCES.items():
        if wanted and key not in wanted:
            continue
        print(f"{key}:")
        try:
            manifest[key] = fetch_hf(key, spec, args.force)
        except Exception as exc:
            # A guardrail corpus being unreachable should degrade that one
            # eval, not abort the whole fetch.
            print(f"  [warn] {key} unavailable: {exc}")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "note": (
            "Every dataset below is public and externally authored. This repo "
            "generates no training, evaluation, or corpus data of its own."
        ),
        "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "datasets": manifest,
    }, indent=2) + "\n")

    print("\nSummary")
    for key, entry in manifest.items():
        stats = entry.get("stats", {})
        detail = stats.get("rows") or stats.get("questions") or "?"
        print(f"  {key:<20} {detail!s:>8} records  {_human(entry['bytes']):>10}  {entry['licence']}")
    print(f"\nManifest written to {MANIFEST.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
