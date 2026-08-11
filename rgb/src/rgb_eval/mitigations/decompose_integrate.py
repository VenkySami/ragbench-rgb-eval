"""Mitigation for Information Integration: query decomposition + merge.

The paper's error analysis (Merging / Ignoring / Misalignment errors) points to
a single cause: the model can't reliably recognize and juggle multiple
sub-questions at once. The paper itself suggests chain-of-thought decomposition
as one possible direction (noting it costs extra latency). We implement the
simplest version: ask the model to split the question into sub-questions,
answer each one independently against the same retrieved context, then merge
the sub-answers into one final answer.
"""
from __future__ import annotations

from .. import prompts
from ..llm_clients import BaseLLMClient


def decompose_and_answer(client: BaseLLMClient, query: str, docs: list[str]) -> str:
    sub_questions_raw = client.generate(prompts.DECOMPOSE_SYSTEM, query)
    sub_questions = [q.strip("-* \t") for q in sub_questions_raw.splitlines() if q.strip()]
    if not sub_questions:
        sub_questions = [query]

    sub_answers = []
    for sub_q in sub_questions:
        system, user = prompts.build_prompt(sub_q, docs)
        sub_answers.append(client.generate(system, user))

    merge_user = (
        f"Original question:\n{query}\n\n"
        + "\n".join(f"Sub-question: {q}\nSub-answer: {a}" for q, a in zip(sub_questions, sub_answers))
    )
    return client.generate(prompts.MERGE_SYSTEM, merge_user)
