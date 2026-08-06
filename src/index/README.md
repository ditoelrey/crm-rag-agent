# Vector index — OpenAI embeddings → local Qdrant

Dense and hybrid retrieval over the 5,923-block CRM corpus, wired directly into
the eval harness so every change is measured against the BM25 baseline.

Run everything from `src/`.

```bash
python -m index.cli build --estimate      # cost first
python -m index.cli build                 # embed + upsert (incremental)
python -m index.cli info
python -m index.cli search "Колку чини потврда за тековна состојба?" --type tariffs
```

```bash
python -m eval.cli run --retriever index.qdrant_retriever:build --tag dense --baseline eval/reports/baseline_bm25.json
```

## Setup

`build` needs `OPENAI_API_KEY` in the environment. Set it yourself — don't paste
a key into a chat or a source file:

```bash
setx OPENAI_API_KEY "sk-..."
```

`setx` persists it for future processes (reopen the shell). For one session only,
use `$env:OPENAI_API_KEY = "sk-..."` in PowerShell.

Full corpus ≈ **1.35M tokens ≈ $0.03** on `text-embedding-3-small`. Re-runs are
free unless block text changed.

## Design

**Embedding cache.** Every vector is persisted in `.cache/embeddings.sqlite3`,
keyed by `sha256(model|dims|text)`. A crashed build resumes for free, a corpus
rebuild only pays for changed blocks, and `--recreate` re-upserts without
re-embedding. The cache lives *outside* `qdrant_data/` on purpose: the index is
disposable, the vectors cost money.

**Point identity.** `uuid5(NAMESPACE, chunk_id)` — deterministic, so upserts
land in place. Paired with a `text_sha256` payload field, builds are incremental
and stale points (blocks removed from the corpus) are pruned.

**`variation_scope` payload.** A service-scoped shared block (terminology,
legalBasis, description, FAQ) is relevant to *every* sibling variation; a
variation-scoped block only to its own. Filtering on `id_variation` alone would
silently drop the shared content that answers half the questions. `variation_scope`
flattens both cases into one list, so "everything relevant to variation 11116"
is a single `MatchAny`:

```bash
python -m index.cli search "Кои документи се потребни?" --variation 11116 --type documents
```

**Manifest.** `qdrant_data/crm_services.manifest.json` records model, dims,
embedded field and corpus sha. The retriever refuses to run if the query
embedder doesn't match the index — a query embedded with a different model is
not a worse search, it's a meaningless one — and warns if the corpus hash moved.
Changing model or dims auto-rebuilds rather than mixing two vector spaces.

**Batching and backoff.** Requests are bounded by input count *and* total
characters. A 400 (over-long batch) is recovered by splitting in half; rate
limits and 5xx get exponential backoff with jitter on top of the SDK's retries.

## Retrievers

| module | what it does |
|---|---|
| `qdrant_retriever.py` | dense cosine search + payload filters; `.search(query, k)` |
| `hybrid.py` | RRF fusion of BM25 + dense |

RRF is the right fusion here because the two scores aren't comparable (unbounded
BM25 term weights vs cosine similarity) — ranks are:

```
score(d) = w_lex / (K + rank_lex(d)) + w_vec / (K + rank_vec(d))      K = 60
```

Each arm is queried to `depth=50` so a document ranked ~30 by one arm can still
be rescued by the other. With `depth == k` fusion degenerates to an intersection.

When a filter is passed, it is applied to the BM25 arm too (in Python) — otherwise
unfiltered lexical hits leak into the fusion.

## Alias layer (`aliases.py`)

Users and the registry use different words. Measured in the corpus:

| user says | occurrences | registry says | occurrences |
|---|---|---|---|
| машина | **0** | подвижни предмети | 430 |
| гасење | **0** | бришење | 1081 |
| фирма | 192 | субјект | 1680 |
| регистрација | 360 | упис | 2235 |
| цена | 153 | тарифа | 490 |

Two sources, different in kind. **Derived**: the registry defines its own
abbreviations in glossary blocks (`Термин: Акционерско друштво (АД)`), so 15 of
them — АД, ДОО, ПДОО, ТП, ЈТД, КД, КДА, СИЗ… — are parsed out and stay current
on rebuild. **Curated**: the gaps above, each justified by counting both words,
because an alias is only worth adding if the target is what the text says.

Expansion **fuses** rather than replaces — the same RRF as the hybrid retriever,
for the same reason (no comparable score scale between the two result sets).
Replacing was measured and rejected: it lifted the underspecified family
(+0.050 ndcg@10) but regressed seven others and overall, because a query that
already names its service gets diluted by synonyms. Fusing keeps what the
original got right and lets the expansion add only what it missed.

Glossary abbreviations are used for matching and displaying legal forms, **not**
for query expansion — expanding them injected "Трговец – поединец" into queries
containing the variation label "Подружница на странско друштво и странски ТП".

A/B against the same gold set:

```bash
python -m eval.cli run --retriever index.hybrid:build_aliased --tag hybrid_alias --baseline eval/reports/hybrid.json
```

On the lexical arm (`index.aliases:build_lexical_aliased`, no API key needed):
overall +0.002, **no regression beyond tolerance**, variation_tariffs +0.050,
curated_tariffs +0.033, curated_process +0.016, curated_underspecified +0.012.

## Structured section fetch (`structured.py`)

Semantic search cannot rank `Број на чекор: 4 | ЦРРСМ врши обработка на
пријавата…` for "како да регистрирам залог" — they share no vocabulary, so no
alias can bridge them. But once the **service** is known, and routing is the
reliable part of this system (`service_acc@1` ≈ 1.000), the rows are a lookup:
`corpus.of_type(2063, "process", 11187)` returns all six, in order, always.

So: resolve (service, variation) from what the retriever did return, map the
question to a section type (`intent.py`), fetch that section, **fuse** it with
the semantic results by RRF. Fusing rather than substituting means a wrong guess
costs ranking slots instead of replacing a correct answer with a confident wrong
one.

Four containment rules, each of which returns *nothing* rather than something:

| rule | why |
|---|---|
| no readable intent → no fetch | injecting the wrong section beats no section only in the model's confidence, not its accuracy |
| no attributable service → no fetch | — |
| multi-variation service with no pinned form → no fetch | АД's fee is 0 МКД and most siblings' are 2452; the agent asks instead |
| a variation needs **two** agreeing hits | one hit is exactly the sibling-confusion this corpus punishes, and fetch amplifies it — trusting a single hit measured `variation_documents` hit@10 **0.846 → 0.615** |

Plus a budget: injection may take at most half the requested slots (`k//2 + 1`),
so a wrong service attribution can't flood the context. The agent over-fetches
(`k × 4`), so it still receives complete sections — the cap only binds at small k.

Intent cues are ordered **most specific first**, and that order is the
precedence: questions trip several at once ("Каде и како го подигнувам
документот" hits documents, process *and* access), and injecting all three
floods the answer with wrong sections. Vague interrogatives (како / каде / кога)
sit last. `пријав` is deliberately *not* a process cue — it's a noun in dozens of
service names.

### Agent directory

The same machinery serves the authorised-agent lists (`scope: "directory"`,
146 blocks, 60 municipalities, 3,010 agents). "Кои се агентите во Гостивар?" is a
directory lookup, not a search, so `fetch_directory` resolves the municipality
and returns its blocks — both lists, kept separate because they carry different
scopes of authority.

Two guards: a municipality alone is not a request (a service question can
mention one, so an agent cue is also required), and `СКОПЈЕ` expands through
`MUNICIPALITY_GROUPS` to all ten Skopje municipalities — 42–50% of all agents
live there, and the literal bucket alone would return a third of them and call
it the list.

Blocks are ordered by part number across every (list, municipality) pair, so a
budget that can't fit everything returns the *first* part of each list rather
than nine parts of one.

### Measured (lexical arm, 532 cases)

```bash
python -m eval.cli run --retriever index.structured:build_lexical --tag struct_lex
python -m eval.cli run --retriever index.structured:build --tag struct        # full stack
```

Overall ndcg@10 **0.663 → 0.753**, hit@1 **0.323 → 0.600**, hit@10 **0.669 → 0.789**.
Twenty families improved; the biggest are `curated_process` **+0.514**
(hit@10 0.000 → 1.000), `intent_process` +0.326, `curated_deadlines` +0.297,
`variation_documents` +0.280, `variation_forms` +0.282.

Two families move down, both explained and neither a quality loss:

- `faq` −0.056 — **hit@10 is unchanged at 0.988**. The FAQ is still in context;
  it just isn't rank 1 now that the actual tariff and deadline rows sit above
  it. For a question like "Колку чини оваа услуга?" those rows are the better
  answer than an FAQ saying "се наплаќа согласно Тарифата". This is a leaky
  family by construction.
- `service_lookup` −0.012 — its gold is the description block alone by design
  (see eval/README), so anything ranked above it scores as a loss.

## `--dry-run`

`HashEmbedder` is a deterministic random-projection bag-of-words over the same MK
tokenizer as BM25. No key, no network, no cost. It exists so the collection,
payload, filters, retriever adapter and eval wiring can be verified before any
billed call — a plumbing bug shows up as a zero, not as "the model is bad".

```bash
python -m index.cli build --dry-run
```

## Known upstream bug (worked around)

qdrant-client **1.18.0 local mode**: `delete_collection()` drops the collection
from its in-memory registry without closing that collection's sqlite handle, so
the on-disk vector storage survives. A later `create_collection()` with a
different vector size then fails with:

```
ValueError: could not broadcast input array from shape (128,) into shape (256,)
```

Reproducible in six lines against a bare client. `reset_storage()` purges the
storage directory before any client is opened, which side-steps it — safe
because the embedding cache lives elsewhere.

Local mode also takes an exclusive lock on `qdrant_data/`: one client at a time,
so don't run `index.cli search` and `eval.cli run` concurrently.

## Files

| file | role |
|---|---|
| `config.py` | model, dims, paths, batch limits, pricing |
| `embedder.py` | `OpenAIEmbedder` (cache + batching + backoff), `HashEmbedder`, `EmbeddingCache` |
| `qdrant_indexer.py` | collection lifecycle, payload mapping, incremental upsert, manifest |
| `qdrant_retriever.py` | dense retriever + filters (eval-compatible) |
| `hybrid.py` | RRF fusion with BM25 |
| `cli.py` | `build` / `info` / `search` |
