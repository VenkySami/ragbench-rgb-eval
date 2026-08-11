"""Answer/refusal/factual-error matching helpers.

Hand-written (no code reused from the reference repo). Uses simple, transparent
substring matching on normalized text, mirroring the paper's own exact-match
philosophy for ED/CR/Rej (as opposed to the LLM-judged Rej*/ED* variants, which
this project explicitly excludes per task instructions).
"""
from __future__ import annotations

import re

from . import config


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _alias_lists(answer) -> list[list[str]]:
    """`answer` can be a single alias-list, a list of alias-lists (info
    integration, one per sub-question), or a plain string (counterfactual)."""
    if isinstance(answer, str):
        return [[answer]]
    if not answer:
        return []
    if isinstance(answer[0], list):
        return answer
    return [answer]


def contains_answer(response: str, answer) -> bool:
    """True if response contains at least one alias for every sub-answer group."""
    norm_response = normalize(response)
    groups = _alias_lists(answer)
    if not groups:
        return False
    return all(
        any(normalize(alias) in norm_response for alias in group)
        for group in groups
    )


def contains_any_answer(response: str, answer) -> bool:
    """True if response contains at least one alias from at least one group
    (used for partial-credit style checks, not the main accuracy metric)."""
    norm_response = normalize(response)
    groups = _alias_lists(answer)
    return any(
        any(normalize(alias) in norm_response for alias in group)
        for group in groups
    )


def is_refusal(response: str) -> bool:
    return normalize(config.REFUSAL_PHRASE) in normalize(response)


def flags_factual_error(response: str) -> bool:
    return normalize(config.FACTUAL_ERROR_PHRASE) in normalize(response)
