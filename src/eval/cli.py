"""
cli.py  --  command line for the eval harness.   Run from src/.
===============================================================

  python -m eval.cli gen                       # freeze the synthetic gold set
  python -m eval.cli run --retriever bm25      # score a retriever, save a report
  python -m eval.cli run --retriever mypkg.index:build --tag dense
  python -m eval.cli report reports/bm25.json --worst 20
  python -m eval.cli compare reports/bm25.json reports/dense.json
  python -m eval.cli selftest                  # verify the harness itself

Plugging in the real index later means one factory function:

    # src/index/dense.py
    def build(corpus):            # corpus: eval.corpus.Corpus
        return MyRetriever(...)   # .name + .search(query, k) -> [chunk_id | Hit]

then `--retriever index.dense:build`. Nothing in the harness changes.
"""
from __future__ import annotations

import argparse

from agent.agent import DEFAULT_MODEL as AGENT_DEFAULT_MODEL
import importlib
import os
import sys

from . import curated as curated_mod
from . import goldset
from .baselines import BM25Retriever, OracleRetriever, RandomRetriever
from .corpus import DEFAULT_CORPUS, Corpus
from .corpus import load as load_corpus
from .goldset import CASES_DIR, EvalCase
from .harness import (DEFAULT_KS, HEADLINE, Report, compare, diff_cases,
                      evaluate, render, worst_cases)

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
SYNTHETIC_PATH = os.path.join(CASES_DIR, "synthetic.jsonl")
CURATED_PATH = os.path.join(CASES_DIR, "curated.jsonl")


def _load_cases(args, c: Corpus) -> list[EvalCase]:
    """Load frozen cases; regenerate the synthetic half on the fly if it was
    never frozen, so a fresh clone can run immediately."""
    cases: list[EvalCase] = []
    if args.cases:
        for path in args.cases:
            cases += goldset.read_cases(path)
    else:
        if args.source in ("all", "synthetic"):
            if os.path.exists(SYNTHETIC_PATH):
                cases += goldset.read_cases(SYNTHETIC_PATH)
            else:
                print(f"note: {os.path.relpath(SYNTHETIC_PATH)} not frozen yet; "
                      f"generating in memory (run `gen` to freeze).", file=sys.stderr)
                cases += goldset.generate(c, seed=args.seed)
        if args.source in ("all", "curated"):
            cases += (goldset.read_cases(CURATED_PATH) if os.path.exists(CURATED_PATH)
                      else curated_mod.expand(c))

    if args.exclude_leaky:
        cases = [x for x in cases if x.family not in goldset.LEAKY_FAMILIES]
    if args.family:
        wanted = set(args.family)
        cases = [x for x in cases if x.family in wanted]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        sys.exit("no cases selected -- check --source / --family / --cases")
    goldset.validate_cases(cases, c)
    if not getattr(args, "keep_ungraded", False):
        # Abstain cases carry no retrieval gold on purpose; scoring them here
        # would report 0.000 for a system doing exactly the right thing.
        graded = [x for x in cases if any(g >= 2 for g in x.gold.values())]
        if len(graded) != len(cases):
            print(f"note: skipping {len(cases) - len(graded)} case(s) with no "
                  f"retrieval gold (expect_behavior=abstain)", file=sys.stderr)
        cases = graded
    return cases


def _build_retriever(spec: str, c: Corpus, cases):
    """`bm25` / `random` / `oracle` / `oracle+noise`, or `module.path:factory`."""
    if spec == "bm25":
        return BM25Retriever(c)
    if spec == "bm25-nostem":
        return BM25Retriever(c, do_stem=False)
    if spec == "bm25-content":
        return BM25Retriever(c, field="content")
    if spec == "random":
        return RandomRetriever(c)
    if spec == "oracle":
        return OracleRetriever(cases)
    if spec == "oracle+noise":
        return OracleRetriever(cases, noise=3, corpus=c)
    if ":" not in spec:
        sys.exit(f"unknown retriever {spec!r}; use a builtin or 'module.path:factory'")
    mod_name, _, factory_name = spec.partition(":")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        sys.exit(f"cannot import {mod_name!r}: {e}")
    factory = getattr(mod, factory_name, None)
    if factory is None:
        sys.exit(f"{mod_name!r} has no attribute {factory_name!r}")
    r = factory(c)
    if not hasattr(r, "search") or not hasattr(r, "name"):
        sys.exit(f"{spec} returned {r!r}, which lacks .search()/.name")
    return r


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_gen(args) -> int:
    c = load_corpus(args.corpus)
    cases = goldset.generate(c, seed=args.seed)
    goldset.validate_cases(cases, c)
    goldset.write_cases(cases, args.out)
    by_family: dict[str, int] = {}
    for case in cases:
        by_family[case.family] = by_family.get(case.family, 0) + 1
    print(f"wrote {len(cases)} synthetic cases -> {args.out}")
    print(f"corpus {os.path.basename(c.path)}  sha={c.sha256[:12]}  blocks={len(c)}")
    for family, n in sorted(by_family.items(), key=lambda kv: -kv[1]):
        print(f"  {family:<28} {n:>4}")
    # Curated cases are resolved from the hand-authored specs in curated.py, so
    # a corpus rebuild that renumbers rows updates the gold instead of rotting it.
    curated = curated_mod.expand(c)
    goldset.validate_cases(curated, c)
    goldset.write_cases(curated, CURATED_PATH)
    print(f"wrote {len(curated)} curated cases  -> {CURATED_PATH}")
    return 0


def cmd_run(args) -> int:
    c = load_corpus(args.corpus)
    cases = _load_cases(args, c)
    retriever = _build_retriever(args.retriever, c, cases)
    report = evaluate(retriever, cases, c, ks=args.k or DEFAULT_KS,
                      progress=args.progress)
    print(render(report))
    out = args.out or os.path.join(
        REPORTS_DIR, f"{args.tag or _slug(retriever.name)}.json")
    report.save(out)
    print(f"\nreport -> {os.path.relpath(out)}")

    if args.baseline:
        base = Report.load(args.baseline)
        regressions = compare(base, report, metric=args.metric,
                              tolerance=args.tolerance)
        if regressions:
            print(f"\nREGRESSION vs {os.path.basename(args.baseline)} "
                  f"({args.metric}, tolerance {args.tolerance}):")
            for r in regressions:
                print(f"  {r}")
            return 1
        print(f"\nno regression vs {os.path.basename(args.baseline)} "
              f"({args.metric}, tolerance {args.tolerance})")
    return 0


def cmd_report(args) -> int:
    report = Report.load(args.path)
    print(render(report))
    if args.worst:
        print(f"\nWORST {args.worst} CASES by {args.metric}"
              f"{f' in {args.family}' if args.family else ''}:")
        for r in worst_cases(report, args.worst, args.metric, args.family):
            print(f"  {r.scores.get(args.metric, 0):.3f}  [{r.family}] {r.query[:88]}")
            print(f"         got: {', '.join(r.top[:3]) or '(nothing)'}")
    return 0


def cmd_compare(args) -> int:
    base, cand = Report.load(args.baseline), Report.load(args.candidate)
    if base.corpus_sha256 != cand.corpus_sha256:
        print(f"warning: different corpora "
              f"({base.corpus_sha256[:12]} vs {cand.corpus_sha256[:12]}) -- "
              f"scores are not strictly comparable", file=sys.stderr)
    print(f"baseline : {base.retriever}  ({base.n_cases} cases)")
    print(f"candidate: {cand.retriever}  ({cand.n_cases} cases)\n")
    width = max(len(f) for f in {*base.by_family, "overall"})
    header = f"{'scope':<{width + 2}}{'baseline':>12}{'candidate':>12}{'delta':>10}"
    print(header)
    print("-" * len(header))
    rows = [("overall", base.overall, cand.overall)]
    rows += [(f, base.by_family[f], cand.by_family.get(f, {})) for f in sorted(base.by_family)]
    for name, brow, crow in rows:
        if args.metric not in brow or args.metric not in crow:
            continue
        b, cv = brow[args.metric], crow[args.metric]
        print(f"{name:<{width + 2}}{b:>12.3f}{cv:>12.3f}{cv - b:>+10.3f}")

    regressions = compare(base, cand, metric=args.metric, tolerance=args.tolerance)
    if args.cases:
        print(f"\nBIGGEST PER-CASE DROPS ({args.metric}):")
        for case_id, b, cv, query in diff_cases(base, cand, metric=args.metric, n=args.cases):
            if cv < b:
                print(f"  {b:.3f} -> {cv:.3f}  {query[:80]}")
    if regressions:
        print(f"\nFAIL: {len(regressions)} regression(s) beyond tolerance {args.tolerance}")
        return 1
    print(f"\nPASS: no regression beyond tolerance {args.tolerance}")
    return 0


def cmd_answer(args) -> int:
    """Score what the agent SAYS. Generation and scoring are separate steps so a
    scorer change costs nothing to re-apply (`--from`)."""
    from . import answers as A

    c = load_corpus(args.corpus)
    cases = _load_cases(args, c)

    if args.from_file:
        records = A.read_answers(args.from_file)
        name = f"replay({os.path.basename(args.from_file)})"
        print(f"re-scoring {len(records)} saved answer(s) from "
              f"{os.path.relpath(args.from_file)}")
    else:
        from agent.agent import CRMAgent
        try:
            agent = CRMAgent(corpus=c, model=args.model, k=args.k,
                             timeout=args.timeout)
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 2
        name = f"{args.model} + {agent.retriever.name}"
        print(f"generating {len(cases)} answer(s) with {args.model} ...")
        try:
            records = A.generate(agent, cases, progress=not args.quiet)
        finally:
            agent.close()
        out_answers = args.save or os.path.join(A.ANSWERS_DIR, f"{args.tag}.jsonl")
        A.write_answers(records, out_answers)
        print(f"answers -> {os.path.relpath(out_answers)}")

    judge_scores = {}
    if args.judge:
        import os as _os

        from openai import OpenAI
        client = OpenAI(api_key=_os.environ.get("OPENAI_API_KEY"))
        print(f"judging groundedness with {args.judge_model} ...")
        by_id = {x.case_id: x for x in cases}
        for r in records:
            if r.case_id in by_id:
                sc, _ = A.judge_groundedness(r, c, client, args.judge_model)
                if sc is not None:
                    judge_scores[r.case_id] = sc

    report = A.score(records, cases, c, retriever_name=name,
                     judge_scores=judge_scores)
    metrics = list(A.SUMMARY_METRICS)
    if judge_scores:
        metrics.insert(-1, "judge_groundedness")
    print()
    print(render(report, metrics=metrics))

    # Counts, not means: _aggregate averages a metric over the cases where it is
    # defined, so every behaviour would read exactly 1.000.
    behaved: dict[str, int] = {}
    for case_result in report.cases:
        for key in case_result.diagnostics:
            if key.startswith("behaved_"):
                label = key.removeprefix("behaved_")
                behaved[label] = behaved.get(label, 0) + 1
    if behaved:
        print("\nBEHAVIOUR OBSERVED: " + "  ".join(
            f"{k}={v}" for k, v in sorted(behaved.items())))
    print(f"COST: ${report.overall.get('cost_usd_total', 0):.4f}   "
          f"tokens={int(report.overall.get('tokens_total', 0)):,}")

    out = args.out or os.path.join(REPORTS_DIR, f"{args.tag}.json")
    report.save(out)
    print(f"report  -> {os.path.relpath(out)}")

    if args.worst:
        print(f"\nWORST {args.worst} BY {args.metric}:")
        for r in worst_cases(report, args.worst, args.metric):
            print(f"  {r.scores.get(args.metric, 0):.3f}  [{r.family}] {r.query[:70]}")

    if args.baseline:
        base = Report.load(args.baseline)
        regressions = compare(base, report, metric=args.metric,
                              tolerance=args.tolerance, min_family_cases=1)
        if regressions:
            print(f"\nREGRESSION vs {os.path.basename(args.baseline)}:")
            for r in regressions:
                print(f"  {r}")
            return 1
        print(f"\nno regression vs {os.path.basename(args.baseline)}")
    return 0


def cmd_selftest(args) -> int:
    from .selftest import main as selftest_main
    return selftest_main(args.corpus)


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):     # Cyrillic on a cp1252 console
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(prog="eval.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate + freeze the synthetic gold set")
    g.add_argument("--out", default=SYNTHETIC_PATH)
    g.add_argument("--seed", type=int, default=20260728)
    g.set_defaults(func=cmd_gen)

    r = sub.add_parser("run", help="score a retriever")
    r.add_argument("--retriever", default="bm25",
                   help="bm25 | bm25-nostem | bm25-content | random | oracle | "
                        "oracle+noise | module.path:factory")
    r.add_argument("--cases", nargs="*", help="explicit case files (overrides --source)")
    r.add_argument("--source", choices=["all", "synthetic", "curated"], default="all")
    r.add_argument("--family", nargs="*", help="only these families")
    r.add_argument("--exclude-leaky", action="store_true",
                   help=f"drop lexically leaky families {sorted(goldset.LEAKY_FAMILIES)}")
    r.add_argument("--limit", type=int)
    r.add_argument("--k", type=int, nargs="*", help=f"cutoffs (default {list(DEFAULT_KS)})")
    r.add_argument("--seed", type=int, default=20260728)
    r.add_argument("--out")
    r.add_argument("--tag", help="report filename stem")
    r.add_argument("--progress", action="store_true")
    r.add_argument("--baseline", help="fail (exit 1) on regression vs this report")
    r.add_argument("--metric", default=HEADLINE)
    r.add_argument("--tolerance", type=float, default=0.01)
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="print a saved report")
    rep.add_argument("path")
    rep.add_argument("--worst", type=int, default=0)
    rep.add_argument("--metric", default=HEADLINE)
    rep.add_argument("--family")
    rep.set_defaults(func=cmd_report)

    cmp_ = sub.add_parser("compare", help="regression gate between two reports")
    cmp_.add_argument("baseline")
    cmp_.add_argument("candidate")
    cmp_.add_argument("--metric", default=HEADLINE)
    cmp_.add_argument("--tolerance", type=float, default=0.01)
    cmp_.add_argument("--cases", type=int, default=0, help="show N biggest per-case drops")
    cmp_.set_defaults(func=cmd_compare)

    a = sub.add_parser("answer", help="score the agent's ANSWERS, not just retrieval")
    a.add_argument("--cases", nargs="*")
    a.add_argument("--source", choices=["all", "synthetic", "curated"],
                   default="curated",
                   help="default 'curated': answer eval costs money per case, and "
                        "the curated set is the honest quality signal anyway")
    a.add_argument("--family", nargs="*")
    a.add_argument("--exclude-leaky", action="store_true")
    a.add_argument("--limit", type=int)
    a.add_argument("--seed", type=int, default=20260728)
    # Follows the agent's default so the eval measures what ships; a copy
    # here silently kept scoring gpt-4o-mini after the default changed.
    a.add_argument("--model", default=AGENT_DEFAULT_MODEL)
    a.add_argument("-k", type=int, default=10)
    a.add_argument("--tag", default="answers")
    a.add_argument("--timeout", type=float, default=120.0,
                   help="per-request timeout in seconds; a hung case is recorded "
                        "as an error and the run continues")
    a.add_argument("--out", help="report path")
    a.add_argument("--save", help="where to write the generated answers")
    a.add_argument("--from", dest="from_file",
                   help="re-score saved answers instead of calling the model")
    a.add_argument("--judge", action="store_true",
                   help="add an LLM groundedness pass (opt-in: not reproducible)")
    a.add_argument("--judge-model", default="gpt-4o-mini")
    a.add_argument("--worst", type=int, default=0)
    a.add_argument("--quiet", action="store_true")
    a.add_argument("--baseline", help="fail (exit 1) on regression vs this report")
    a.add_argument("--metric", default="answer_score")
    a.add_argument("--tolerance", type=float, default=0.01)
    a.set_defaults(func=cmd_answer, keep_ungraded=True)

    st = sub.add_parser("selftest", help="known-answer tests for the harness itself")
    st.set_defaults(func=cmd_selftest)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
