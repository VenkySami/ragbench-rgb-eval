"""Mitigation for Noise Robustness: lexical-overlap relevance filter.

The paper's error analysis for Noise Robustness (long-distance information,
evidence uncertainty, concept confusion) boils down to one root cause: once
positive and negative documents are mixed, the model has no help telling them
apart. A cheap fix that needs no extra LLM call: score each candidate document's
lexical overlap with the query and drop the least-relevant ones before they ever
reach the answering prompt, shrinking the effective noise ratio the model sees.
"""
from __future__ import annotations

import re

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "for", "to",
    "and", "or", "what", "who", "when", "where", "which", "did", "does", "do",
    "with", "by", "at", "it", "its", "as", "be", "that", "this",
}


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _overlap_score(query_tokens: set[str], doc: str) -> int:
    return len(query_tokens & _tokens(doc))


def filter_relevant_docs(query: str, docs: list[str], keep_fraction: float = 0.6) -> list[str]:
    """Keeps the top `keep_fraction` of docs by lexical overlap with the query
    (at least 1 doc is always kept). Ties keep original order (stable sort)."""
    if not docs:
        return docs
    query_tokens = _tokens(query)
    scored = list(enumerate(docs))
    scored.sort(key=lambda pair: _overlap_score(query_tokens, pair[1]), reverse=True)
    keep_n = max(1, round(len(docs) * keep_fraction))
    kept_indices = sorted(idx for idx, _ in scored[:keep_n])
    return [docs[i] for i in kept_indices]
