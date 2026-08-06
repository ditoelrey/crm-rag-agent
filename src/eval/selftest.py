"""
selftest.py  --  known-answer tests for the harness itself.
===========================================================
An eval harness is the instrument every later decision is read off, so it needs
its own calibration check. Three layers:

  1. metrics    hand-computed nDCG/MRR/coverage values, so a refactor of the
                formulas cannot silently shift every score.
  2. gold set   the validator must reject stale chunk_ids -- the failure mode
                that looks identical to a broken retriever.
  3. end-to-end oracle must score a perfect 1.0 and random must score near 0 on
                the real corpus. If the oracle is not 1.0, the harness is wrong,
                not the retriever.

Run: python -m eval.cli selftest        (from src/)
"""
from __future__ import annotations

import math
import traceback

from . import curated, goldset, metrics as M
from .baselines import BM25Retriever, OracleRetriever, RandomRetriever
from .corpus import DEFAULT_CORPUS, load as load_corpus
from .goldset import EvalCase
from .harness import evaluate
from .text import stem, tokenize

_failures: list[str] = []


def check(label: str, got, want, tol: float = 1e-9) -> None:
    ok = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r}, want {want!r}")
    if not ok:
        _failures.append(label)


def check_true(label: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{(' -- ' + detail) if detail else ''}")
    if not cond:
        _failures.append(label)


def _metrics_tests() -> None:
    print("[metrics]")
    ranked = ["c1", "c2", "c3", "c4"]

    check("hit@1 when rank1 is gold", M.hit_at_k(ranked, {"c1": 2}, 1), 1.0)
    check("hit@1 when rank3 is gold", M.hit_at_k(ranked, {"c3": 2}, 1), 0.0)
    check("hit@3 when rank3 is gold", M.hit_at_k(ranked, {"c3": 2}, 3), 1.0)
    check("grade-1 alone never counts as a hit", M.hit_at_k(ranked, {"c1": 1}, 4), 0.0)

    check("rr for rank 3", M.rr_at_k(ranked, {"c3": 2}, 10), 1 / 3)
    check("rr outside cutoff", M.rr_at_k(ranked, {"c4": 2}, 3), 0.0)

    gold4 = {"a": 2, "b": 2, "c": 2, "d": 2}
    mixed = ["a", "x", "y", "z", "w"]
    check("recall@2 with 4 gold", M.recall_at_k(mixed, gold4, 2), 0.25)
    check("coverage@2 normalizes by reachable", M.coverage_at_k(mixed, gold4, 2), 0.5)
    check("coverage@k>|gold| equals recall", M.coverage_at_k(mixed, gold4, 10), 0.25)
    check("precision@2", M.precision_at_k(mixed, gold4, 2), 0.5)

    # DCG = 0 + 1/log2(3) + 3/log2(4);  IDCG = 3/log2(2) + 1/log2(3)
    dcg = 1 / math.log2(3) + 3 / math.log2(4)
    idcg = 3 / math.log2(2) + 1 / math.log2(3)
    check("ndcg@3 graded", M.ndcg_at_k(["c1", "c2", "c3"], {"c3": 2, "c2": 1}, 3),
          dcg / idcg, tol=1e-9)
    check("ndcg is 1.0 for a perfect ranking",
          M.ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, 2), 1.0)
    check("ndcg is 0.0 when nothing relevant is returned",
          M.ndcg_at_k(["x", "y"], {"a": 2}, 2), 0.0)
    check("empty gold scores 0", M.ndcg_at_k(ranked, {}, 5), 0.0)


def _text_tests() -> None:
    print("[text]")
    check("definite article stripped", stem("документите"), "документ")
    check("short root protected", stem("дом"), "дом")
    check("unstemmed tokenization keeps surface forms",
          tokenize("Колку чини АД регистрација?", do_stem=False),
          ["колку", "чини", "ад", "регистрација"])
    check_true("stopwords dropped", "на" not in tokenize("документи на субјект"))
    check_true("intent word survives stopword filtering",
               "колку" in tokenize("колку чини", do_stem=False))
    check_true("inflected forms merge to one term",
               stem("документи") == stem("документот") == stem("документите"),
               f"{stem('документи')} / {stem('документот')} / {stem('документите')}")


def _goldset_tests(c) -> None:
    print("[goldset]")
    bogus = EvalCase(case_id="x", query="q", family="f", source="curated",
                     gold={"srv_does_not_exist_v0_process_1": 2})
    try:
        goldset.validate_cases([bogus], c)
        check_true("validator rejects a stale gold chunk_id", False)
    except ValueError as e:
        check_true("validator rejects a stale gold chunk_id", "not in corpus" in str(e))

    real_id = c.blocks[0].chunk_id
    no_primary = EvalCase(case_id="y", query="q", family="f", source="curated",
                          gold={real_id: 1})
    try:
        goldset.validate_cases([no_primary], c)
        check_true("validator rejects gold with no primary", False)
    except ValueError as e:
        check_true("validator rejects gold with no primary", "primary" in str(e))

    cases_a = goldset.generate(c, seed=7)
    cases_b = goldset.generate(c, seed=7)
    check_true("generation is deterministic for a fixed seed",
               [x.case_id for x in cases_a] == [x.case_id for x in cases_b] and
               [x.query for x in cases_a] == [x.query for x in cases_b])
    goldset.validate_cases(cases_a, c)
    check_true("generated gold validates", True, f"{len(cases_a)} cases")

    cur = curated.expand(c)
    goldset.validate_cases(cur, c)
    check("every curated spec resolves", len(cur), len(curated.SPECS))
    copy_tolerant = [x for x in cur if x.family.endswith(("faq", "terminology"))]
    check_true("boilerplate gold accepts its duplicate copies",
               all(sum(1 for g in x.gold.values() if g >= 2) > 1 for x in copy_tolerant),
               f"{len(copy_tolerant)} copy-tolerant curated cases")

    var_cases = [x for x in cases_a if x.expect_variation is not None]
    check_true("variation family is populated", len(var_cases) > 20,
               f"{len(var_cases)} variation-targeted cases")
    leaked = [x for x in var_cases
              if any(c.by_id[cid].id_variation not in (None, x.expect_variation)
                     for cid, g in x.gold.items() if g >= 2)]
    check_true("no sibling variation is graded primary", not leaked,
               f"{len(leaked)} leaky cases")


def _end_to_end_tests(c) -> None:
    print("[end-to-end]")
    cases = goldset.generate(c, seed=7)[:120]

    oracle = evaluate(OracleRetriever(cases), cases, c)
    check("oracle hit@1", oracle.overall["hit@1"], 1.0)
    check("oracle mrr@10", oracle.overall["mrr@10"], 1.0)
    check("oracle ndcg@20", oracle.overall["ndcg@20"], 1.0, tol=1e-6)

    noisy = evaluate(OracleRetriever(cases, noise=3, corpus=c), cases, c)
    check_true("noise degrades the oracle", noisy.overall["ndcg@10"] < oracle.overall["ndcg@10"],
               f"{noisy.overall['ndcg@10']:.3f} < {oracle.overall['ndcg@10']:.3f}")

    rnd = evaluate(RandomRetriever(c), cases, c)
    check_true("random is near the floor", rnd.overall["ndcg@10"] < 0.02,
               f"ndcg@10={rnd.overall['ndcg@10']:.4f}")

    bm25 = evaluate(BM25Retriever(c), cases, c)
    check_true("bm25 clears random by a wide margin",
               bm25.overall["ndcg@10"] > rnd.overall["ndcg@10"] + 0.2,
               f"bm25={bm25.overall['ndcg@10']:.3f} random={rnd.overall['ndcg@10']:.3f}")
    check_true("report carries the corpus hash", len(bm25.corpus_sha256) == 64)


def main(corpus_path: str = DEFAULT_CORPUS) -> int:
    _failures.clear()
    c = load_corpus(corpus_path)
    print(f"corpus: {c.path}\n        {len(c)} blocks, sha={c.sha256[:12]}\n")
    for stage in (_metrics_tests, _text_tests):
        stage()
    for stage in (_goldset_tests, _end_to_end_tests):
        try:
            stage(c)
        except Exception:
            traceback.print_exc()
            _failures.append(stage.__name__)
    print()
    if _failures:
        print(f"SELFTEST FAILED: {len(_failures)} check(s): {', '.join(_failures)}")
        return 1
    print("SELFTEST PASSED")
    return 0
