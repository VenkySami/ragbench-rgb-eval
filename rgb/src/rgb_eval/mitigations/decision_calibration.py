"""Mitigation for Negative Rejection: verification-then-answer prompting.

The paper's own finding: LLMs "fail to strictly follow instructions" and
generate unpredictable text instead of a clean refusal when no document
supports the answer. We tighten the system instruction to force an explicit
internal check ("does ANY document support an answer?") before generating
anything else, using `prompts.VERIFY_THEN_ANSWER_SYSTEM`.
"""
from __future__ import annotations

from .. import prompts
from ..llm_clients import BaseLLMClient


def verify_then_answer(client: BaseLLMClient, query: str, docs: list[str]) -> str:
    _, user = prompts.build_prompt(query, docs)
    return client.generate(prompts.VERIFY_THEN_ANSWER_SYSTEM, user)
