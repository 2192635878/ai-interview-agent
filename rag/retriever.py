from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_+#.-]+|[\u4e00-\u9fff]")
DEFAULT_KNOWLEDGE_PATH = Path(__file__).resolve().parents[1] / "data" / "knowledge_base.json"


@dataclass
class KnowledgeChunk:
    title: str
    content: str
    role: str = ""
    tags: List[str] | None = None
    source: str = "local"

    def text(self) -> str:
        return f"{self.title}\n{self.content}"


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


def load_knowledge_chunks(path: Path = DEFAULT_KNOWLEDGE_PATH) -> List[KnowledgeChunk]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    chunks = []
    for item in data:
        chunks.append(
            KnowledgeChunk(
                title=item.get("title", ""),
                content=item.get("content", ""),
                role=item.get("role", ""),
                tags=list(item.get("tags", [])),
                source=item.get("source", "local"),
            )
        )
    return chunks


class KeywordRetriever:
    def __init__(self, chunks: Sequence[KnowledgeChunk]) -> None:
        self.chunks = list(chunks)
        self.chunk_tokens = [tokenize(chunk.text() + " " + " ".join(chunk.tags or [])) for chunk in self.chunks]
        self.document_frequency: Dict[str, int] = {}
        for tokens in self.chunk_tokens:
            for token in set(tokens):
                self.document_frequency[token] = self.document_frequency.get(token, 0) + 1

    def search(
        self,
        query: str,
        role: str = "",
        tags: Iterable[str] | None = None,
        top_k: int = 3,
    ) -> List[Dict[str, object]]:
        if not self.chunks:
            return []

        query_tokens = tokenize(query + " " + role + " " + " ".join(tags or []))
        if not query_tokens:
            return []

        wanted_tags = {tag.lower() for tag in (tags or [])}
        scored = []
        total_docs = len(self.chunks)
        for index, chunk in enumerate(self.chunks):
            chunk_tags = {tag.lower() for tag in (chunk.tags or [])}
            role_bonus = 0.4 if role and (chunk.role == role or not chunk.role) else 0.0
            tag_bonus = 0.25 * len(wanted_tags & chunk_tags)
            score = role_bonus + tag_bonus
            token_counts: Dict[str, int] = {}
            for token in self.chunk_tokens[index]:
                token_counts[token] = token_counts.get(token, 0) + 1
            length_norm = max(len(self.chunk_tokens[index]), 1)
            for token in query_tokens:
                tf = token_counts.get(token, 0)
                if not tf:
                    continue
                df = self.document_frequency.get(token, 0)
                idf = math.log((total_docs + 1) / (df + 1)) + 1
                score += (tf / length_norm) * idf * 8
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "title": chunk.title,
                "content": chunk.content,
                "role": chunk.role,
                "tags": chunk.tags or [],
                "source": chunk.source,
                "score": round(score, 3),
            }
            for score, chunk in scored[:top_k]
        ]
