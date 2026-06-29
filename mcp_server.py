from __future__ import annotations

"""
Optional MCP-ready adapter.

The Streamlit app uses local tools by default so the public demo does not depend on
an external MCP runtime. This file documents the stable tool boundary and can be
wrapped by a real MCP SDK server later.
"""

import json
from pathlib import Path
from typing import Iterable

from agent.tools import retrieve_knowledge_tool, search_question_bank_tool
from rag import KeywordRetriever, load_knowledge_chunks


ROOT = Path(__file__).resolve().parent
QUESTION_BANK_PATH = ROOT / "data" / "question_bank.json"


def _load_question_bank() -> dict:
    with QUESTION_BANK_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def search_question_bank(role: str, difficulty: str, tags: Iterable[str] | None = None, top_k: int = 5) -> dict:
    return search_question_bank_tool(_load_question_bank(), role, difficulty, tags, top_k).to_dict()


def retrieve_knowledge(query: str, role: str = "", tags: Iterable[str] | None = None, top_k: int = 3) -> dict:
    retriever = KeywordRetriever(load_knowledge_chunks())
    return retrieve_knowledge_tool(retriever, query, role, tags, top_k).to_dict()


def get_user_profile_schema() -> dict:
    return {
        "client_id": "string",
        "preferred_role": "string",
        "average_score": "number",
        "weaknesses": ["string"],
        "strengths": ["string"],
        "review_plan": ["string"],
    }


if __name__ == "__main__":
    print("MCP-ready tools: search_question_bank, retrieve_knowledge, get_user_profile_schema")
