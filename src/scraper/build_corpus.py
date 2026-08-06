"""
build_corpus.py  --  the single entry point that produces data/jsonl/corpus.jsonl.
==================================================================================
Two sources, one corpus:

  data/raw/*.json          archived service payloads  -> crm_parser
  data/attachments/*.xlsx  authorised-agent lists     -> agents_parser

Writes the JSONL and then runs the strict validator, so a malformed block can
never reach the index: build and validate are one step, not two that can drift.

  python build_corpus.py                 # from src/scraper
  python build_corpus.py --no-agents     # services only
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import agents_parser
import crm_parser
from validate_corpus import validate_file

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "data", "raw")
ATTACH_DIR = os.path.join(HERE, "data", "attachments")
OUT_PATH = os.path.join(HERE, "data", "jsonl", "corpus.jsonl")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default=RAW_DIR)
    p.add_argument("--attachments", default=ATTACH_DIR)
    p.add_argument("--out", default=OUT_PATH)
    p.add_argument("--no-agents", action="store_true")
    args = p.parse_args(argv)

    result = crm_parser.build_corpus(args.raw)
    blocks = result["blocks"]
    print("services : " + "  ".join(f"{k}={v}" for k, v in result["stats"].items()))
    if result["skipped"]:
        print(f"  skipped {len(result['skipped'])} unusable payload(s)")

    if not args.no_agents and os.path.isdir(args.attachments):
        agent_blocks = agents_parser.build_all(args.attachments)
        if agent_blocks:
            by_list = Counter(b["metadata"]["list"] for b in agent_blocks)
            total = sum(b["metadata"]["n_agents"] for b in agent_blocks
                        if b["metadata"]["part"] == 1)
            municipalities = len({(b["metadata"]["list"], b["metadata"]["municipality"])
                                  for b in agent_blocks})
            print(f"agents   : {len(agent_blocks)} blocks  {dict(by_list)}  "
                  f"{municipalities} list/municipality groups  {total} agents")
            blocks = blocks + agent_blocks

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n = crm_parser.write_jsonl(blocks, args.out)
    print(f"wrote {n} blocks -> {os.path.relpath(args.out, HERE)}")

    report = validate_file(args.out)
    print("VALIDATION PASSED")
    for k, v in report.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
