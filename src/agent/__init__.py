"""Interactive RAG agent over the CRM corpus.

  python -m agent.cli
  python -m agent.cli -q "Колку чини потврда за тековна состојба?"
"""
from __future__ import annotations

from .agent import Answer, CRMAgent

__all__ = ["CRMAgent", "Answer", "context", "prompts"]
