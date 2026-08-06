# RAG agent

Hybrid retrieval (k=10) → XML context → OpenAI → grounded Macedonian answer with
citations. Run from `src/`.

```bash
python -m agent.cli
python -m agent.cli -q "Колку чини потврда за тековна состојба на фирма?"
python -m agent.cli -q "..." --show-context
python -m agent.selftest          # offline, no API key, no cost
```

REPL: `/sources` `/context` `/reset` `/cost` `/quit`.
Flags: `--model` `-k` `--temperature` `--no-dedup`.

Defaults: `gpt-4o-mini`, `temperature=0.0`, `k=10`, dedup on.
Local Qdrant holds an exclusive lock on `qdrant_data/` — don't run this
concurrently with `index.cli search` or `eval.cli run`.

## Shape of a turn

```
user message
  └─ plan retrieval   (merge a pending clarification; derive a variation filter)
  └─ hybrid search    (BM25 ⊕ dense, RRF, k=10)
  └─ build context    (dedup identical blocks, detect variation ambiguity)
  └─ OpenAI           (system rules ‖ history ‖ system documents ‖ user turn)
  └─ validate         (every [citation] must exist in the retrieved set)
     → Answer
```

`ask()` returns an `Answer` dataclass — text, docs, valid/invalid citations,
ambiguity, tokens, cost, split latencies — not a string. Answer-quality eval will
reuse `src/eval`'s runner and regression gate, and prose can't be scored.

## Three behaviours

**Citations.** Every factual claim carries the exact `chunk_id` in brackets.
They're also *verified* after generation: any bracketed id not in the retrieved
set is reported as fabricated and the CLI marks the answer unverified. A model
that invents a source id is a hard failure for a legal-answer system, and
instructing it not to isn't the same as checking.

**Abstention.** No answer outside the documents. The prompt names the specific
trap this corpus sets: a near-miss service (documents about *changing* a pledge
when the user asked about *registering* one) must be declined, not repurposed.

**Variation disambiguation.** Detected deterministically, not left to the model,
by two detectors — strongest first:

- **evidence** — two or more legal forms of the same service are already in the
  retrieved documents, contributing the same section type.
- **structure** — the corpus schema says the attributed service has sibling
  variations whose relevant section *differs*, even if retrieval surfaced none
  of them.

The structure detector is the one that matters. On the bare question "Колку чини
регистрација?" retrieval returned **zero tariff rows** — "регистрација" matches
blocks about registering a user account in the e-system, while the fee rows say
"упис на основање". An evidence-only detector stays silent exactly when the user
most needs asking, on a question where АД and ПДОО cost 0 МКД and the other seven
forms cost 2452 МКД. That exact retrieved set is a fixture in `selftest.py`.

Three guards keep it from over-firing, each tested in both directions:

| guard | why |
|---|---|
| service attributed only from blocks with `id_variation > 0` | shared terminology/FAQ text is boilerplate replicated across services and says nothing about *which* service is meant. The rank-1 hit in the failure belonged to 2155, which has no variations at all; the block that identified the subject sat at rank 4. |
| the section must genuinely differ across forms | all five variations of "Поднесување годишна сметка" share one identical access block — nothing to choose, so nothing to ask. |
| the question must have a readable intent | `detect_intent` maps MK cues to a section type; no intent means no guess about which section would differ. Cues match on a left word boundary so "чин" doesn't fire inside "начин". |

The follow-up is what makes it useful: when the user replies "АД", the next turn
retrieves on *original question + reply* (the reply alone retrieves nothing) and
converts the named form into a `variation_scope` filter, which keeps the
service-level shared blocks that apply to every form. Both deterministic, no
extra model call.

## Deduplication

16–20% of every top-10 was byte-identical text repeated under different services,
and ~20% of queries wasted four or more of ten slots on the same sentence
(measured across the three retriever reports). 88% of `process` rows, 93% of
`terminology` and 94% of `documentsLocations` appear verbatim under more than one
service — the portal replicates boilerplate.

Collapsing duplicates recovers ~2 usable slots out of 10 at zero cost. Provenance
survives: the copy kept is the best-ranked one, and the other services it covers
go into `also_applies_to` on the document tag, so citations still point somewhere
checkable. `--no-dedup` turns it off for comparison.

## Files

| file | role |
|---|---|
| `context.py` | dedup, ambiguity detection, XML rendering, citation validation |
| `prompts.py` | system prompt (English rules, Macedonian output) |
| `agent.py` | `CRMAgent.ask()` → `Answer`; multi-turn clarification |
| `cli.py` | REPL + one-shot |
| `selftest.py` | 30 offline checks: dedup, ambiguity, citations, orchestration |
