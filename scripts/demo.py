#!/usr/bin/env python3
"""Walk the whole system in one run: ask, attack, and inspect.

    python3 scripts/demo.py
    python3 scripts/demo.py --ask "What is the Amazon rainforest?"

Every question below is a real SQuAD question drawn from the corpus at
runtime, and every attack is a real entry from a published attack dataset.
Nothing is typed in by hand, so what you see is what the evaluation measures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundwork.config import SETTINGS  # noqa: E402
from groundwork.gateway.router import Gateway  # noqa: E402
from groundwork.rag.corpus import load_questions, stratified_sample  # noqa: E402
from groundwork.rag.index import HybridIndex  # noqa: E402
from groundwork.rag.pipeline import RagPipeline  # noqa: E402
from groundwork.security.authz import KeyStore, generate_key  # noqa: E402

RULE = "=" * 74


def banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def show(answer, expectation: str = "") -> None:
    print(f"\nQ: {answer.question}")
    if expectation:
        print(f"   (reference: {expectation})")
    print(f"A: {answer.text}")
    print(f"   decision={answer.decision.value}  confidence={answer.confidence:.3f}  "
          f"groundedness={answer.groundedness:.2f}")
    if answer.citations:
        sources = {p.citation_tag: p.title for p in answer.passages}
        cited = ", ".join(f"{tag} = {sources.get(tag, '?')}" for tag in answer.citations)
        print(f"   citations: {cited}")
    if answer.blocked_by:
        print(f"   stopped by: {', '.join(answer.blocked_by)}")
    if answer.rail_notes:
        print(f"   rails: {'; '.join(answer.rail_notes)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ask", help="ask a single question and exit")
    args = parser.parse_args()

    index = HybridIndex.load()

    # A real keystore, built at runtime. The gateway refuses unauthenticated
    # calls, so even the demo authenticates.
    keystore = KeyStore()
    demo_key = generate_key()
    keystore.add(demo_key, "analytics")

    pipeline = RagPipeline(index, gateway=Gateway(keystore=keystore))
    principal = pipeline.gateway.authenticate(demo_key)

    print(f"Corpus: {len(index)} real Wikipedia passages (SQuAD v2.0 dev)")
    print(f"Provider: {SETTINGS.provider}")
    print(f"Authenticated as tenant '{principal.tenant}' (key id {principal.key_id})")

    if args.ask:
        show(pipeline.answer(args.ask, tenant=principal.tenant))
        return 0

    questions = load_questions()

    # -- answerable -------------------------------------------------------
    banner("1. Questions the corpus can answer")
    print("Real SQuAD questions whose gold answer is present in the corpus.")
    for question in [q for q in stratified_sample(questions, 400, seed=7) if q.answerable][:3]:
        show(pipeline.answer(question.question, tenant=principal.tenant),
             expectation=question.answers[0] if question.answers else "")

    # -- unanswerable -----------------------------------------------------
    banner("2. Questions the corpus cannot answer")
    print("Real SQuAD v2 unanswerable questions. Crowdworkers wrote these to look")
    print("answerable from the paragraph while having no answer in it, which is")
    print("exactly the case where a RAG system confabulates.")
    for question in [q for q in stratified_sample(questions, 400, seed=7) if not q.answerable][:3]:
        show(pipeline.answer(question.question, tenant=principal.tenant),
             expectation="no answer exists")

    # -- attacks ----------------------------------------------------------
    banner("3. Real attacks from published corpora")
    try:
        from groundwork.evaluation.datasets import load_gandalf, load_prompt_injections

        attacks = [(t, "deepset/prompt-injections")
                   for t, l in zip(load_prompt_injections().texts, load_prompt_injections().labels)
                   if l == 1][:2]
        attacks += [(t, "Lakera/gandalf") for t in load_gandalf().texts[:2]]
        for text, source in attacks:
            print(f"\n[{source}]")
            show(pipeline.answer(text[:400], tenant=principal.tenant))
    except FileNotFoundError:
        print("  attack corpora not fetched; run scripts/fetch_data.py")

    # -- gateway ----------------------------------------------------------
    banner("4. Gateway accounting")
    stats = pipeline.gateway.stats()
    print(f"provider: {stats['provider']}")
    print(f"cache:    {stats['cache']}")
    for tenant in stats["tenants"]:
        print(f"tenant:   {tenant}")

    banner("5. Audit trail")
    ok, checked, bad = pipeline.gateway.audit.verify()
    print(f"hash chain verified: {ok} over {checked} records"
          + (f" (first bad: {bad})" if bad else ""))
    print("Records carry event, tenant, outcome, cost, and token counts.")
    print("They deliberately carry no prompt text, no retrieved passages, and")
    print("no model output: logging those would recreate the exposure the")
    print("guardrails exist to prevent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
