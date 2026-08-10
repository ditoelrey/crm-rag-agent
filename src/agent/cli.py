"""
cli.py  --  interactive chat with the CRM agent.   Run from src/.
=================================================================

  python -m agent.cli                                  # REPL
  python -m agent.cli -q "Колку чини потврда за тековна состојба?"
  python -m agent.cli -q "..." --show-context          # dump the XML the model saw
  python -m agent.cli --model gpt-4o --k 20

REPL commands:  /sources  /context  /reset  /cost  /help  /quit

Local Qdrant takes an exclusive lock on qdrant_data/, so don't run this at the
same time as `index.cli search` or `eval.cli run`.
"""
from __future__ import annotations

import argparse
import sys

from eval.corpus import DEFAULT_CORPUS
from eval.corpus import load as load_corpus

from .agent import DEFAULT_K, DEFAULT_MODEL, DEFAULT_TEMPERATURE, Answer, CRMAgent
from .context import build_documents, render_documents

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def _print_answer(ans: Answer, *, verbose: bool = True) -> None:
    print(f"\n{ans.text}\n")
    if ans.invalid_citations:
        print(f"{BOLD}!! fabricated citation(s): "
              f"{', '.join(ans.invalid_citations)}{RESET}")
        print(f"{DIM}   Not in the retrieved set -- treat this answer as "
              f"unverified.{RESET}")
    elif ans.stale_citations:
        print(f"{DIM}[cites {', '.join(ans.stale_citations)} from an earlier "
              f"turn -- shown before, but not re-retrieved for this answer]{RESET}")
    elif not ans.citations and not ans.ambiguity:
        # Not an error: an abstention has nothing to cite. It IS worth showing,
        # because the other way to get here is an answer whose claims came from
        # a document it declined to name.
        print(f"{DIM}[no citations -- abstention, or claims not tied to a "
              f"source]{RESET}")
    if not verbose:
        return
    if ans.ambiguity:
        print(f"{DIM}[clarification requested: "
              f"{', '.join(ans.ambiguity.labels)}]{RESET}")
    if ans.filters:
        print(f"{DIM}[filtered retrieval: {ans.filters}]{RESET}")
    print(f"{DIM}{len(ans.citations)}/{len(ans.docs)} documents cited  |  "
          f"retrieval {ans.retrieval_ms:.0f} ms + generation {ans.generation_ms:.0f} ms  |  "
          f"{ans.prompt_tokens}+{ans.completion_tokens} tok  ${ans.cost_usd:.5f}{RESET}")


def _print_sources(ans: Answer) -> None:
    if not ans.docs:
        print("no documents retrieved")
        return
    cited = set(ans.citations)
    for d in ans.docs:
        mark = "*" if d.chunk_id in cited else " "
        b = d.block
        label = b.service_name[:56] + (f" — {b.variation_short_name}"
                                       if b.variation_short_name else "")
        print(f" {mark} {d.rank:>2}. [{b.type}] {label}")
        print(f"{DIM}      {d.chunk_id}   score={d.score:.4f}"
              f"{('   also: ' + '; '.join(d.also_in[:2])) if d.also_in else ''}{RESET}")
    print(f"{DIM} * = cited in the answer{RESET}")


def _one_shot(agent: CRMAgent, query: str, show_context: bool) -> int:
    ans = agent.ask(query)
    if show_context:
        print(render_documents(ans.docs))
        print()
    _print_answer(ans)
    _print_sources(ans)
    return 0


def _repl(agent: CRMAgent) -> int:
    print(f"{BOLD}Централен регистар — RAG агент{RESET}")
    print(f"{DIM}model={agent.model}  k={agent.k}  temp={agent.temperature}  "
          f"retriever={agent.retriever.name[:60]}{RESET}")
    print(f"{DIM}/sources /context /reset /cost /quit{RESET}\n")

    last: Answer | None = None
    while True:
        try:
            line = input(f"{BOLD}> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd = line.lower()
            if cmd in ("/quit", "/exit", "/q"):
                break
            if cmd == "/reset":
                agent.reset()
                last = None
                print(f"{DIM}conversation cleared{RESET}")
            elif cmd == "/sources":
                if last:
                    _print_sources(last)
                else:
                    print("nothing asked yet")
            elif cmd == "/context":
                if last:
                    print(render_documents(last.docs))
                else:
                    print("nothing asked yet")
            elif cmd == "/cost":
                print(f"session total: ${agent.total_cost_usd:.5f}")
            elif cmd in ("/help", "/h"):
                print("/sources  /context  /reset  /cost  /quit")
            else:
                print(f"unknown command {line}")
            continue

        try:
            last = agent.ask(line)
        except Exception as e:                      # keep the session alive
            print(f"{BOLD}error: {type(e).__name__}: {e}{RESET}")
            continue
        _print_answer(last)

    print(f"{DIM}session cost ${agent.total_cost_usd:.5f}{RESET}")
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(prog="agent.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-q", "--query", help="ask one question and exit")
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("-k", type=int, default=DEFAULT_K, help="documents retrieved")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--timeout", type=float, default=120.0,
                   help="per-request timeout in seconds")
    p.add_argument("--no-dedup", action="store_true",
                   help="keep byte-identical duplicate blocks in the context")
    p.add_argument("--no-aliases", action="store_true",
                   help="disable user-vocabulary query expansion")
    p.add_argument("--no-structured", action="store_true",
                   help="disable deterministic section fetch")
    p.add_argument("--show-context", action="store_true",
                   help="print the XML the model was given (one-shot mode)")
    args = p.parse_args(argv)

    corpus = load_corpus(args.corpus)
    try:
        agent = CRMAgent(corpus=corpus, model=args.model, k=args.k,
                         temperature=args.temperature, dedup=not args.no_dedup,
                         aliases=not args.no_aliases,
                         structured=not args.no_structured,
                         timeout=args.timeout)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2
    try:
        return (_one_shot(agent, args.query, args.show_context) if args.query
                else _repl(agent))
    finally:
        agent.close()


if __name__ == "__main__":
    raise SystemExit(main())
