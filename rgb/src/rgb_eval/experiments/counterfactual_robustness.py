"""Counterfactual Robustness experiment: Error Detection Rate (ED) and Error
Correction Rate (CR) on en_fact.json, with an optional mitigation
(parametric-knowledge cross-check).
"""
from __future__ import annotations

from .. import matching, metrics, prompts
from ..llm_clients import BaseLLMClient
from ..mitigations.fact_check import cross_check_answer


def run(client: BaseLLMClient, sample_size: int | None = None, mitigation: bool = False) -> list[dict]:
    from .. import data_loader
    records = data_loader.load_records("counterfactual_robustness")
    records = data_loader.sample_records(records, sample_size)

    flags, corrected = [], []
    for rec in records:
        docs = rec.extra.get("positive_wrong") or rec.positive  # counterfactual documents
        if mitigation:
            response = cross_check_answer(client, rec.query, docs)
        else:
            system, user = prompts.build_prompt(rec.query, docs)
            response = client.generate(system, user)
        flags.append(matching.flags_factual_error(response))
        corrected.append(matching.contains_answer(response, rec.answer))

    return [{
        "ability": "counterfactual_robustness",
        "model": client.name,
        "mitigation": mitigation,
        "n": len(flags),
        "error_detection_rate": metrics.error_detection_rate(flags),
        "error_correction_rate": metrics.error_correction_rate(flags, corrected),
    }]
