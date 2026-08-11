"""Mitigation for Counterfactual Robustness: parametric-vs-retrieved cross-check.

The paper's own conclusion: "existing LLMs do not have a safeguard... they
heavily depend on the information they retrieve" even when their internal
knowledge is correct. We add a safeguard: first ask the model for its own
no-context ("parametric") answer, then explicitly ask it to compare that answer
against the retrieved documents and resolve any conflict, instead of answering
from the documents alone.
"""
from __future__ import annotations

from .. import prompts
from ..llm_clients import BaseLLMClient


def cross_check_answer(client: BaseLLMClient, query: str, docs: list[str]) -> str:
    own_answer = client.generate(prompts.PARAMETRIC_ONLY_SYSTEM, query)
    user = prompts.CROSSCHECK_USER_TEMPLATE.format(
        query=query,
        own_answer=own_answer,
        docs=prompts.format_docs(docs),
    )
    return client.generate(prompts.CROSSCHECK_SYSTEM, user)
