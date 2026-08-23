"""
answers.py  --  scoring what the agent SAYS, not just what it retrieved.
========================================================================
The retrieval harness scores what reaches the context window. It cannot see what
the model does with it -- and every answer-quality defect in this project so far
was found by a human reading output:

  * a fully-cited four-step procedure where the registry defines six, silently
    dropping the processing, approval and refusal steps;
  * one of six required documents listed under a heading implying the full set;
  * a lawyer welded together from a Струмица first name and a Гостивар address,
    assembled out of stale conversation history, citing nothing;
  * a citation pointing at part 1 of a list for a fact that lives in part 2.

None of those moved a retrieval metric. All of them are the kind of thing a user
acts on.

Design: DETERMINISTIC CORE, OPTIONAL JUDGE
------------------------------------------
An LLM judge is expensive, non-reproducible, and unnecessary for most of what
matters here. A fee is right or it is not; a deadline is present or it is not.
So the default gate is entirely deterministic and free:

  behaviour_match       answer / clarify / abstain -- did it do the right KIND
                        of thing? An agent that always answers scores well on
                        every other metric while being unsafe.
  value_recall          the checkable facts the answer must contain (295 МКД,
                        15 дена, 4 часа) -- verified against the corpus when the
                        case was written.
  no_forbidden          the near-miss value must NOT appear: 299 МКД is a real
                        tariff for a different certificate, 5140 МКД a real
                        price for a smaller tender package. Quoting one for the
                        other is the most expensive mistake this system can make.
  numeric_groundedness  every multi-digit number in the answer must appear
                        somewhere in the retrieved context. Fees, deadlines,
                        counts and phone numbers are exactly where fabrication
                        does damage, and they are checkable without a judge.
  value_citation        for each expected value the answer states, at least one
                        CITED block must actually contain it. This is the check
                        that catches "cited part 1 for a fact in part 2".
  citation_integrity    no fabricated ids.

`--judge` adds a model-scored groundedness pass for prose claims that carry no
number. It is opt-in precisely because it cannot be a stable regression gate.

Generation and scoring are separate steps. `generate()` writes answers to JSONL;
`score()` reads them back. Iterating on a scorer therefore costs nothing, and a
run can be re-graded after the rules change without paying for the model again.
"""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from .corpus import Corpus
from .goldset import EvalCase
from .harness import CaseResult, Report, _aggregate, _group, _p95, _relpath

ANSWERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "answers")
HEADLINE = "answer_score"

# Macedonian phrasings that mark a refusal. The prompt suggests the first; the
# others are what the model actually produces in practice.
_ABSTAIN_MARKERS = (
    "нема информација", "не располагам", "не се достапни", "не е достапна",
    "не е достапно", "не содржи", "не располага", "нема податоци",
    "не можам да најдам", "не постои информација",
)

# A clarifying question, as distinct from an answer that happens to end in "?".
_CLARIFY_MARKERS = (
    "за каква правна форма", "која опција", "кој вариjант", "кој вид",
    "ве молам да ми кажете", "ве молам наведете", "на кој начин сакате",
    "за кој субјект", "прецизирајте",
)

# Numbers worth checking: two or more digits, so enumeration markers ("1.",
# "2.") and single-digit counts are not treated as claims.
_NUMBER_RE = re.compile(r"\d[\d.,/\-]*\d|\d{2,}")
_LIST_MARKER_RE = re.compile(r"^\s*\**\s*\d+[.)]\s", re.M)
# chunk_ids are full of digits (srv_2162_v11050_tariffs_1). Counting those as
# factual claims made a perfectly grounded answer score 0.33.
_CITATION_RE = re.compile(r"\[[^\]]*\](?:\([^)]*\))?")


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
@dataclass
class AnswerRecord:
    """One generated answer, persisted so scoring can be re-run for free."""
    case_id: str
    query: str
    text: str
    citations: list[str] = field(default_factory=list)
    stale_citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    docs: list[str] = field(default_factory=list)
    asked_clarification: bool = False
    ambiguity_source: str | None = None
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    error: str | None = None
    # Multi-turn: one entry per turn. `text`/`citations` above are the FINAL
    # answer, so every single-turn scorer keeps working unchanged; `docs` is the
    # union across turns, because a fact retrieved on turn 1 is legitimately
    # available on turn 3.
    turns: list[dict] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "AnswerRecord":
        known = set(AnswerRecord.__dataclass_fields__)
        return AnswerRecord(**{k: v for k, v in d.items() if k in known})


def generate(agent, cases: Sequence[EvalCase], *, progress: bool = True
             ) -> list[AnswerRecord]:
    """Run the agent over the cases, each in a FRESH conversation.

    Resetting BETWEEN cases is correctness, not hygiene: conversation history is
    what produced the fabricated lawyer, so letting it bleed across evaluation
    cases would score a system nobody runs. Within a case the history is kept,
    because that is the thing under test.
    """
    out: list[AnswerRecord] = []
    for i, case in enumerate(cases, 1):
        agent.reset()
        turns: list[dict] = []
        docs: list[str] = []
        last = None
        error = None
        try:
            for message in case.conversation:
                a = agent.ask(message)
                last = a
                turns.append({
                    "query": message,
                    "text": a.text,
                    "citations": list(a.citations),
                    "docs": [d.chunk_id for d in a.docs],
                    "asked_clarification": a.asked_for_clarification,
                })
                for d in a.docs:
                    if d.chunk_id not in docs:
                        docs.append(d.chunk_id)
        except Exception as e:                     # never lose a whole run
            error = f"{type(e).__name__}: {e}"

        if last is None:
            out.append(AnswerRecord(case_id=case.case_id, query=case.query,
                                    text="", error=error, turns=turns))
        else:
            out.append(AnswerRecord(
                case_id=case.case_id, query=case.query, text=last.text,
                citations=list(last.citations),
                stale_citations=list(last.stale_citations),
                invalid_citations=list(last.invalid_citations),
                docs=docs, asked_clarification=last.asked_for_clarification,
                ambiguity_source=last.ambiguity.source if last.ambiguity else None,
                model=last.model,
                prompt_tokens=sum(t.get("prompt_tokens", 0) for t in turns) or last.prompt_tokens,
                completion_tokens=last.completion_tokens,
                cost_usd=agent.total_cost_usd - sum(r.cost_usd for r in out),
                retrieval_ms=last.retrieval_ms, generation_ms=last.generation_ms,
                error=error, turns=turns))
        if progress:
            spent = sum(r.cost_usd for r in out)
            n = f" ({len(case.conversation)} turns)" if case.is_multi_turn else ""
            print(f"  {i}/{len(cases)}  ${spent:.4f}  {case.query[:48]}{n}", flush=True)
    return out


def write_answers(records: Sequence[AnswerRecord], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    return path


def read_answers(path: str) -> list[AnswerRecord]:
    with open(path, encoding="utf-8") as fh:
        return [AnswerRecord.from_dict(json.loads(line))
                for line in fh if line.strip()]


# --------------------------------------------------------------------------- #
# deterministic scorers
# --------------------------------------------------------------------------- #
def observed_behavior(record: AnswerRecord) -> str:
    """What the agent actually did: answer | clarify | abstain.

    Read from the TEXT, not from the ambiguity flag. The flag records what the
    context asked the model to do; this metric grades what the model did. A live
    run handed the model an ambiguity block, the model correctly ignored it and
    answered with the link -- and grading it "clarify" from the flag measured our
    detector rather than the agent.

    Both a refusal and a clarification are recognised by what they LACK: neither
    states a fact or cites a source. An answer that gives the fee and then notes
    that one detail is missing is an ANSWER; grading it otherwise rewards hedging.
    """
    return classify(record.text, record.citations, record.asked_clarification)


def classify(text: str, citations: Sequence[str] = (),
             asked_clarification: bool = False) -> str:
    """Behaviour of one answer. Shared by the whole-case and per-turn scorers."""
    t = _norm(text)
    if not t:
        return "error"
    if bool(citations) or bool(numbers_in(text)):
        return "answer"
    if any(m in t for m in _ABSTAIN_MARKERS):
        return "abstain"
    if asked_clarification or any(m in t for m in _CLARIFY_MARKERS):
        return "clarify"
    return "answer"


def value_recall(record: AnswerRecord, case: EvalCase) -> float | None:
    """Share of the expected facts the answer states.

    A value may list equivalent renderings separated by "|", and any one counts.
    The corpus writes a date as "15.3."; a good answer may write "15 март". An
    earlier version demanded the literal corpus string and scored a clearer,
    factually identical answer at 0.0 -- rewarding regurgitation over
    communication, which is the opposite of what this metric is for.
    """
    if not case.expect_values:
        return None
    text = _norm(record.text)
    found = sum(1 for v in case.expect_values
                if any(_norm(alt) in text for alt in v.split("|")))
    return found / len(case.expect_values)


def _states_unnegated(text: str, needle: str) -> bool:
    """Is `needle` present at least once WITHOUT a preceding "не"?

    Forbidding a claim means forbidding the assertion, not the words. Macedonian
    negates by placing "не" directly before the verb, so the forbidden phrase is
    a literal substring of its own denial: an answer that correctly says "агентот
    НЕ може да ви ја среди документацијата" contains "може да ви ја среди". The
    first version of this check scored exactly that answer 0.0 -- marking the
    model wrong for finally getting it right.

    Every occurrence is examined, so "агентот може ..., но не може ..." still
    trips on the first, unnegated one.
    """
    start = 0
    while (i := text.find(needle, start)) >= 0:
        before = text[:i].rstrip()
        if not (before.endswith("не") and (len(before) == 2 or not before[-3].isalpha())):
            return True
        start = i + 1
    return False


def forbidden_absent(record: AnswerRecord, case: EvalCase) -> float | None:
    """1.0 when no near-miss value appears. The wrong fee is a real fee.

    Like `value_recall`, an entry may list equivalent phrasings separated by
    "|", and ANY of them trips it. Without this the field only catches wording
    it was written against: the NGO false-premise case forbade "агентот може"
    and happily passed an answer that said "регистрационен агент може да ви
    помогне" -- the same false claim, scored 0.75.
    """
    if not case.forbid_values:
        return None
    text = _norm(record.text)
    return 0.0 if any(_states_unnegated(text, _norm(alt))
                      for v in case.forbid_values
                      for alt in v.split("|")) else 1.0


def numbers_in(text: str) -> list[str]:
    """Multi-digit numbers stated as claims, ignoring list markers and citations."""
    stripped = _LIST_MARKER_RE.sub("", _CITATION_RE.sub(" ", text))
    out = []
    for m in _NUMBER_RE.findall(stripped):
        d = _digits(m)
        if len(d) >= 2:
            out.append(d)
    return out


def numeric_groundedness(record: AnswerRecord, corpus: Corpus) -> float | None:
    """Share of the answer's numbers that appear in the retrieved context.

    A deterministic stand-in for groundedness that targets exactly where
    fabrication costs the user: amounts, deadlines, counts, phone numbers.
    """
    stated = numbers_in(record.text)
    if not stated:
        return None
    pool = " ".join(corpus.by_id[cid].content for cid in record.docs
                    if cid in corpus.by_id)
    available = set(numbers_in(pool))
    # "15.3." normalises to "153", but an answer that renders it "15 март"
    # states "15" -- which IS in the context, just not as one token. Expose the
    # components of separated numbers so a correct paraphrase is not read as a
    # fabrication. Unseparated amounts ("2452") are unaffected, so an invented
    # figure still fails.
    for token in re.findall(r"\d[\d.,/\-]*\d", pool):
        for part in re.split(r"[.,/\-]+", token):
            if len(part) >= 2:
                available.add(part)
    # Dates and years the model may legitimately restate from its own framing.
    grounded = sum(1 for n in stated if n in available)
    return grounded / len(stated)


def value_citation(record: AnswerRecord, case: EvalCase, corpus: Corpus) -> float | None:
    """For each expected value the answer states, does a CITED block contain it?

    Retrieval-level citation checking only asks whether an id was returned. This
    asks whether the id supports the claim -- the gap that let an answer cite
    part 1 of a municipality list for an agent listed in part 2.
    """
    if not case.expect_values:
        return None
    text = _norm(record.text)
    stated = [v for v in case.expect_values if _norm(v) in text]
    if not stated:
        return None
    cited_text = _norm(" ".join(corpus.by_id[cid].content for cid in record.citations
                                if cid in corpus.by_id))
    if not cited_text:
        return 0.0
    return sum(1 for v in stated if _norm(v) in cited_text) / len(stated)


def score_case(record: AnswerRecord, case: EvalCase, corpus: Corpus,
               judge_score: float | None = None) -> tuple[dict[str, float], dict[str, float]]:
    scores: dict[str, float] = {}
    diag: dict[str, float] = {}

    observed = observed_behavior(record)
    diag[f"behaved_{observed}"] = 1.0
    if case.expect_behavior:
        scores["behavior_match"] = 1.0 if observed == case.expect_behavior else 0.0

    if case.expect_turn_behaviors and record.turns:
        # Every turn must do the right KIND of thing, in order. A conversation
        # that ends correctly after asking the wrong question along the way is
        # not the behaviour we shipped.
        want = case.expect_turn_behaviors
        got = [classify(t.get("text", ""), t.get("citations", ()),
                        t.get("asked_clarification", False))
               for t in record.turns]
        matched = sum(1 for w, g in zip(want, got) if w == g)
        scores["turn_behavior_match"] = matched / max(len(want), 1)
        diag["turns"] = float(len(got))

    for name, value in (("value_recall", value_recall(record, case)),
                        ("no_forbidden", forbidden_absent(record, case)),
                        ("numeric_groundedness", numeric_groundedness(record, corpus)),
                        ("value_citation", value_citation(record, case, corpus))):
        if value is not None:
            scores[name] = value

    scores["citation_integrity"] = 0.0 if record.invalid_citations else 1.0
    diag["n_citations"] = float(len(record.citations))
    diag["n_stale_citations"] = float(len(record.stale_citations))
    diag["cited_something"] = 1.0 if record.citations else 0.0
    if judge_score is not None:
        scores["judge_groundedness"] = judge_score

    # Composite headline: the mean of whatever applies to this case. Deliberately
    # unweighted -- a weighted blend invites tuning the weights instead of the
    # system. Read the components; the composite is only for the gate.
    scores[HEADLINE] = sum(scores.values()) / len(scores) if scores else 0.0
    return scores, diag


# --------------------------------------------------------------------------- #
# optional LLM judge
# --------------------------------------------------------------------------- #
JUDGE_PROMPT = """\
You are grading a Macedonian answer produced by a retrieval-augmented assistant.

Decide ONLY this: is every factual claim in the answer supported by the supplied
documents? Do not judge whether the answer is helpful, complete, or well written.
Do not use outside knowledge — if a claim is true in the world but absent from
the documents, it is UNSUPPORTED.

Reply with JSON only:
{"supported": <integer>, "unsupported": <integer>, "examples": ["<short quote>"]}
where the two integers count factual claims in the answer.
"""


def judge_groundedness(record: AnswerRecord, corpus: Corpus, client,
                       model: str = "gpt-4o-mini") -> tuple[float | None, str]:
    if not record.text.strip():
        return None, ""
    docs = "\n\n".join(f"[{cid}] {corpus.by_id[cid].content}"
                       for cid in record.docs if cid in corpus.by_id)
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.0,
            messages=[{"role": "system", "content": JUDGE_PROMPT},
                      {"role": "user",
                       "content": f"DOCUMENTS:\n{docs}\n\nANSWER:\n{record.text}"}])
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        sup, unsup = int(data.get("supported", 0)), int(data.get("unsupported", 0))
        total = sup + unsup
        return (sup / total if total else None), raw
    except Exception as e:
        return None, f"judge failed: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
SUMMARY_METRICS = ("behavior_match", "value_recall", "no_forbidden",
                   "numeric_groundedness", "value_citation", HEADLINE)


def score(records: Sequence[AnswerRecord], cases: Sequence[EvalCase],
          corpus: Corpus, *, retriever_name: str = "agent",
          judge_scores: dict[str, float] | None = None) -> Report:
    by_id = {c.case_id: c for c in cases}
    judge_scores = judge_scores or {}
    results: list[CaseResult] = []

    for r in records:
        case = by_id.get(r.case_id)
        if case is None:
            continue
        scores, diag = score_case(r, case, corpus, judge_scores.get(r.case_id))
        if r.error:
            diag["errors"] = 1.0
        results.append(CaseResult(
            case_id=r.case_id, family=case.family, source=case.source,
            difficulty=case.difficulty, query=r.query, scores=scores,
            diagnostics=diag, top=list(r.citations),
            latency_ms=round(r.retrieval_ms + r.generation_ms, 1),
        ))

    overall = _aggregate(results)
    lat = [x.latency_ms for x in results]
    overall["latency_ms_mean"] = sum(lat) / len(lat) if lat else 0.0
    overall["latency_ms_p95"] = _p95(lat)
    overall["cost_usd_total"] = sum(r.cost_usd for r in records)
    overall["tokens_total"] = float(sum(r.prompt_tokens + r.completion_tokens
                                        for r in records))
    return Report(
        retriever=retriever_name,
        corpus_path=_relpath(corpus.path),
        corpus_sha256=corpus.sha256,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ks=[], n_cases=len(results), overall=overall,
        by_family=_group(results, lambda x: x.family),
        by_source=_group(results, lambda x: x.source),
        by_difficulty=_group(results, lambda x: x.difficulty),
        cases=results,
    )
