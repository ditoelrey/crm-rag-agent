"""
harness.py  --  run a retriever over a gold set and produce a comparable report.
================================================================================
Beyond the usual ranking metrics, the harness computes diagnostics specific to
this corpus, because "recall went down" is not actionable but these are:

  service_acc@k       did the right *service* appear at all? (routing)
  variation_acc@1     for variation-targeted cases: is rank 1 the right
                      variation (or a service-level shared block), rather than a
                      sibling? A sibling hit is a confidently wrong legal answer.
  sibling_confusion@k share of the top k that belongs to the right service but
                      the WRONG variation -- the single most dangerous failure
                      mode this corpus has (АД vs ДОО vs Здружение).
  type_precision@k    share of the top k whose section type matches the intent
                      ("колку чини" -> tariffs). Low values mean the ranker is
                      matching the service name and ignoring the question.

Reports are JSON, stamped with the corpus sha256 and the retriever's `name`, and
`compare()` turns two of them into a pass/fail regression gate.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from . import metrics as M
from .corpus import _SRC_DIR, Corpus
from .goldset import EvalCase
from .retriever import Hit, Retriever, normalize_hits

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10, 20)
HEADLINE = "ndcg@10"


@dataclass
class CaseResult:
    case_id: str
    family: str
    source: str
    difficulty: str
    query: str
    scores: dict[str, float]
    diagnostics: dict[str, float]
    top: list[str]
    latency_ms: float


@dataclass
class Report:
    retriever: str
    corpus_path: str
    corpus_sha256: str
    generated_at: str
    ks: list[int]
    n_cases: int
    overall: dict[str, float]
    by_family: dict[str, dict[str, float]]
    by_source: dict[str, dict[str, float]]
    by_difficulty: dict[str, dict[str, float]]
    cases: list[CaseResult] = field(default_factory=list)

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, ensure_ascii=False, indent=2)
        return path

    @staticmethod
    def load(path: str) -> "Report":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        d["cases"] = [CaseResult(**c) for c in d.get("cases", [])]
        return Report(**d)


# --------------------------------------------------------------------------- #
# Scoring one case
# --------------------------------------------------------------------------- #
def _score_case(case: EvalCase, hits: Sequence[Hit], c: Corpus,
                ks: Sequence[int]) -> tuple[dict[str, float], dict[str, float]]:
    ranked = [h.chunk_id for h in hits]
    scores: dict[str, float] = {}
    for k in ks:
        scores[f"hit@{k}"] = M.hit_at_k(ranked, case.gold, k)
        scores[f"recall@{k}"] = M.recall_at_k(ranked, case.gold, k)
        scores[f"coverage@{k}"] = M.coverage_at_k(ranked, case.gold, k)
        scores[f"precision@{k}"] = M.precision_at_k(ranked, case.gold, k)
        scores[f"ndcg@{k}"] = M.ndcg_at_k(ranked, case.gold, k)
    scores["mrr@10"] = M.rr_at_k(ranked, case.gold, 10)

    diag: dict[str, float] = {}
    blocks = [c.by_id[cid] for cid in ranked]

    if case.expect_types:
        want = set(case.expect_types)
        for k in ks:
            top = blocks[:k]
            diag[f"type_precision@{k}"] = (
                sum(1 for b in top if b.type in want) / len(top) if top else 0.0)

    if case.expect_service is not None:
        sid = case.expect_service
        for k in ks:
            diag[f"service_acc@{k}"] = 1.0 if any(b.id_service == sid for b in blocks[:k]) else 0.0

        if case.expect_variation is not None:
            vid = case.expect_variation
            # rank 1 is "right" if it is the target variation, or a service-level
            # block that legitimately applies to every sibling.
            if blocks:
                b0 = blocks[0]
                diag["variation_acc@1"] = 1.0 if (
                    b0.id_service == sid and (b0.is_shared or b0.id_variation == vid)) else 0.0
            else:
                diag["variation_acc@1"] = 0.0
            for k in ks:
                top = blocks[:k]
                if not top:
                    diag[f"sibling_confusion@{k}"] = 0.0
                    continue
                diag[f"sibling_confusion@{k}"] = sum(
                    1 for b in top
                    if b.id_service == sid and not b.is_shared and b.id_variation != vid
                ) / len(top)
    return scores, diag


def _aggregate(results: Iterable[CaseResult]) -> dict[str, float]:
    """Mean per metric, averaged only over the cases where it is defined --
    variation diagnostics must not be diluted by cases that cannot have them."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    n = 0
    for r in results:
        n += 1
        for name, value in (*r.scores.items(), *r.diagnostics.items()):
            sums[name] = sums.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1
    out = {name: sums[name] / counts[name] for name in sorted(sums)}
    out["n_cases"] = float(n)
    return out


def _group(results: Sequence[CaseResult], key) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[CaseResult]] = {}
    for r in results:
        buckets.setdefault(key(r), []).append(r)
    return {name: _aggregate(rows) for name, rows in sorted(buckets.items())}


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
def evaluate(retriever: Retriever, cases: Sequence[EvalCase], c: Corpus, *,
             ks: Sequence[int] = DEFAULT_KS, keep_top: int = 10,
             progress: bool = False) -> Report:
    ks = sorted(set(ks))
    max_k = max(ks)
    results: list[CaseResult] = []

    for i, case in enumerate(cases, 1):
        t0 = time.perf_counter()
        raw = retriever.search(case.query, max_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        hits = normalize_hits(raw, max_k)

        for h in hits:
            if h.chunk_id not in c.by_id:
                raise ValueError(
                    f"retriever {retriever.name!r} returned chunk_id {h.chunk_id!r} "
                    f"for case {case.case_id!r}, which is not in {c.path}. "
                    f"The index is stale or built from a different corpus.")

        scores, diag = _score_case(case, hits, c, ks)
        results.append(CaseResult(
            case_id=case.case_id, family=case.family, source=case.source,
            difficulty=case.difficulty, query=case.query, scores=scores,
            diagnostics=diag, top=[h.chunk_id for h in hits[:keep_top]],
            latency_ms=round(latency_ms, 3),
        ))
        if progress and i % 50 == 0:
            print(f"  ... {i}/{len(cases)} cases", flush=True)

    overall = _aggregate(results)
    overall["latency_ms_mean"] = M.mean([r.latency_ms for r in results])
    overall["latency_ms_p95"] = _p95([r.latency_ms for r in results])

    return Report(
        retriever=retriever.name,
        corpus_path=_relpath(c.path),
        corpus_sha256=c.sha256,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ks=list(ks), n_cases=len(results), overall=overall,
        by_family=_group(results, lambda r: r.family),
        by_source=_group(results, lambda r: r.source),
        by_difficulty=_group(results, lambda r: r.difficulty),
        cases=results,
    )


def _relpath(path: str) -> str:
    """Report the corpus location relative to src/ so reports compare across
    machines; fall back to the raw path if it lives outside the tree."""
    try:
        return os.path.relpath(path, _SRC_DIR).replace(os.sep, "/")
    except ValueError:
        return path


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


# --------------------------------------------------------------------------- #
# Presentation + regression gate
# --------------------------------------------------------------------------- #
_SUMMARY_METRICS = ("hit@1", "hit@5", "coverage@5", "mrr@10", "ndcg@10")


def render(report: Report, *, metrics: Sequence[str] = _SUMMARY_METRICS) -> str:
    lines = [
        f"retriever : {report.retriever}",
        f"corpus    : {report.corpus_path}  sha={report.corpus_sha256[:12]}",
        f"cases     : {report.n_cases}   generated {report.generated_at}",
        f"latency   : mean {report.overall.get('latency_ms_mean', 0):.1f} ms  "
        f"p95 {report.overall.get('latency_ms_p95', 0):.1f} ms",
        "",
    ]
    lines += _table("OVERALL", {"all": report.overall}, metrics)
    lines += [""] + _table("BY SOURCE", report.by_source, metrics)
    lines += [""] + _table("BY FAMILY", report.by_family, metrics)

    diag_names = [n for n in ("service_acc@1", "service_acc@5", "type_precision@5",
                              "variation_acc@1", "sibling_confusion@5")
                  if any(n in row for row in report.by_family.values())]
    if diag_names:
        lines += [""] + _table("DIAGNOSTICS BY FAMILY", report.by_family, diag_names)
    return "\n".join(lines)


def _table(title: str, rows: dict[str, dict[str, float]],
           metrics: Sequence[str]) -> list[str]:
    label_w = max([len(title), *(len(k) for k in rows)] or [len(title)]) + 2
    head = f"{title:<{label_w}}" + "".join(f"{m:>18}" for m in metrics) + f"{'n':>7}"
    out = [head, "-" * len(head)]
    for name, row in rows.items():
        cells = "".join(
            (f"{row[m]:>18.3f}" if m in row else f"{'-':>18}") for m in metrics)
        out.append(f"{name:<{label_w}}{cells}{int(row.get('n_cases', 0)):>7}")
    return out


@dataclass
class Regression:
    scope: str          # "overall" or "family:<name>"
    metric: str
    baseline: float
    candidate: float

    @property
    def delta(self) -> float:
        return self.candidate - self.baseline

    def __str__(self) -> str:
        return (f"{self.scope:<28} {self.metric:<12} "
                f"{self.baseline:.3f} -> {self.candidate:.3f}  ({self.delta:+.3f})")


def compare(baseline: Report, candidate: Report, *, metric: str = HEADLINE,
            tolerance: float = 0.01, min_family_cases: int = 10) -> list[Regression]:
    """Regressions worse than `tolerance`, overall and per family.

    Per-family checks are the point: a change that lifts the easy leaky families
    while quietly destroying `variation_documents` is a net loss disguised as a
    win, and the overall mean will not show it.
    """
    out: list[Regression] = []
    if metric in baseline.overall and metric in candidate.overall:
        b, c = baseline.overall[metric], candidate.overall[metric]
        if c < b - tolerance:
            out.append(Regression("overall", metric, b, c))
    for family, brow in baseline.by_family.items():
        crow = candidate.by_family.get(family)
        if not crow or metric not in brow or metric not in crow:
            continue
        if min(brow.get("n_cases", 0), crow.get("n_cases", 0)) < min_family_cases:
            continue
        if crow[metric] < brow[metric] - tolerance:
            out.append(Regression(f"family:{family}", metric, brow[metric], crow[metric]))
    return sorted(out, key=lambda r: r.delta)


def worst_cases(report: Report, n: int = 20, metric: str = "ndcg@10",
                family: str | None = None) -> list[CaseResult]:
    rows = [r for r in report.cases if family is None or r.family == family]
    return sorted(rows, key=lambda r: r.scores.get(metric, 0.0))[:n]


def diff_cases(baseline: Report, candidate: Report, *, metric: str = "ndcg@10",
               n: int = 20) -> list[tuple[str, float, float, str]]:
    """Per-case deltas, biggest drop first -- for finding out *why* a family moved."""
    base = {r.case_id: r for r in baseline.cases}
    rows = []
    for r in candidate.cases:
        b = base.get(r.case_id)
        if b is None:
            continue
        rows.append((r.case_id, b.scores.get(metric, 0.0), r.scores.get(metric, 0.0), r.query))
    rows.sort(key=lambda t: t[2] - t[1])
    return rows[:n]
