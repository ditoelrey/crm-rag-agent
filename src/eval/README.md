# Retrieval eval harness

Scores any retriever over the CRM e-services corpus. Built **before** the index
on purpose: the index is a choice, the measurement is not, and every later
decision (BM25 vs dense vs hybrid, chunk size, reranker, filters) is only
defensible if it moves a number that was defined first.

Pure stdlib. Run everything from `src/`.

```bash
python -m eval.cli selftest                      # verify the harness itself
python -m eval.cli gen                           # freeze the gold set
python -m eval.cli run --retriever bm25 --tag baseline_bm25
python -m eval.cli report eval/reports/baseline_bm25.json --worst 20
python -m eval.cli compare eval/reports/baseline_bm25.json eval/reports/dense.json
```

## Plugging in your index

One function. Nothing in the harness changes.

```python
# src/index/dense.py
def build(corpus):                 # corpus: eval.corpus.Corpus
    return MyRetriever(corpus)     # .name  and  .search(query, k) -> [chunk_id | Hit]
```

```bash
python -m eval.cli run --retriever index.dense:build --tag dense \
       --baseline eval/reports/baseline_bm25.json
```

Contract: at most `k` results, best first, no duplicates, and every returned
`chunk_id` must exist in the corpus — the harness raises on a stale id rather
than scoring it as a miss, because "your index is out of date" and "your ranker
is bad" must not look the same.

## The two tiers of gold

| tier | n | what it is | what it's for |
|---|---|---|---|
| synthetic | 512 | generated from the corpus (`goldset.py`) | regression signal, full coverage, deterministic |
| curated | 23 | hand-written questions (`curated.py`) | the only honest quality signal |

Cases also carry `expect_behavior` (`answer` / `clarify` / `abstain`) — what the
*agent* should do, as opposed to what retrieval should find. Retrieval metrics
ignore it; the answer eval will not. A `clarify` case still carries retrieval
gold, because the agent can only ask an informed question if the competing rows
were retrievable in the first place.

The `curated_underspecified` family (n=5) is the one to watch: questions with an
intent but no named service, which is how people actually ask. BM25 scores
**0.000 hit@5 / 0.031 ndcg@10** on it.

The gap between them is the point. BM25 scores **0.669 ndcg@10** on synthetic and
**0.480** on curated — synthetic families quote corpus text back at the retriever
(an FAQ question appears verbatim inside its own answer block), so they read
high. Use `--exclude-leaky` to drop the `faq` and `terminology` families, and
never quote a synthetic number as system quality.

Curated gold lives as *rules* (`service` + `variation` + section type), not
frozen ids, so rebuilding the corpus re-resolves it. Add cases by appending a
`Spec` to `curated.SPECS` — 15 honest cases beat 500 generated ones, and this
file should grow every time a real question is seen to fail.

### Grades

`2` primary (answers the query) · `1` acceptable (same service/variation
context) · absent = irrelevant. recall/hit/MRR count only grade 2; nDCG uses
both. Sibling variations are never graded — that is what makes confusion
measurable.

## Metrics

`hit@k` `recall@k` `precision@k` `mrr@10` `ndcg@k`, plus `coverage@k` =
`|hit| / min(|gold|, k)`, which is what to read when gold sets differ in size
(recall@5 is capped at 0.42 when 12 document rows are all correct).

Diagnostics say *why* a number moved:

- **`service_acc@k`** — routing: did the right service appear at all?
- **`variation_acc@1`** — is rank 1 the right variation (or a shared block), not
  a sibling? АД vs ДОО vs Здружение have different documents and different fees,
  so a sibling hit is a confidently wrong legal answer.
- **`sibling_confusion@k`** — share of top-k from the right service, wrong variation.
- **`type_precision@k`** — share of top-k whose section type matches the intent.
  Low = the ranker is matching the service name and ignoring the question.

## Baseline (BM25, corpus sha `4903bbd7`)

| slice | hit@1 | hit@5 | mrr@10 | ndcg@10 | n |
|---|---|---|---|---|---|
| all | 0.326 | 0.567 | 0.423 | 0.669 | 527 |
| curated | 0.333 | 0.600 | 0.439 | 0.480 | 15 |

Two findings the index has to answer for:

1. **Routing is solved, intent is not.** `service_acc@1` is 0.95–1.00 almost
   everywhere, but `type_precision@5` is 0.05–0.35: the contextual header puts
   the service name in every block, so "колку чини X" reliably finds *service X*
   and then returns its FAQs instead of its `tariffs` row.
2. **Stemming is load-bearing.** `--retriever bm25-nostem` regresses 6 families
   (variation_process −0.044, variation_deadlines −0.041); MK definite-article
   suffixes are not optional for this corpus.

## Regression gate

`run --baseline <report.json>` exits 1 on a drop beyond `--tolerance` (default
0.01), overall *or per family*. Per-family is the reason it exists: a change
that lifts the easy families while destroying `variation_documents` is a net
loss that the overall mean hides.

## Files

| file | role |
|---|---|
| `corpus.py` | typed read-only view + lookup tables + content-duplicate index |
| `text.py` | MK tokenizer/stemmer — shared with the future lexical index |
| `metrics.py` | ranking metrics, pure functions |
| `goldset.py` | synthetic case generation, case I/O, gold validation |
| `curated.py` | hand-authored specs |
| `retriever.py` | the `search(query, k)` protocol |
| `baselines.py` | BM25 / random floor / oracle ceiling |
| `harness.py` | scoring, aggregation, rendering, `compare()` |
| `selftest.py` | known-answer tests for the harness itself |
| `cli.py` | `gen` `run` `report` `compare` `selftest` |
