"""Negative Rejection experiment: rejection rate at noise_rate=1.0 on
en_refine.json, with an optional mitigation (verify-then-answer prompt).
"""
from __future__ import annotations

from .. import config, data_loader, matching, metrics, prompts
from ..llm_clients import BaseLLMClient
from ..mitigations.decision_calibration import verify_then_answer


def run(client: BaseLLMClient, sample_size: int | None = None, mitigation: bool = False) -> list[dict]:
    records = data_loader.load_records("negative_rejection")
    records = data_loader.sample_records(records, sample_size)

    refusals = []
    for rec in records:
        docs = data_loader.sample_docs(rec, config.NEGATIVE_REJECTION_NOISE_RATE, config.PASSAGE_NUM)
        if mitigation:
            response = verify_then_answer(client, rec.query, docs)
        else:
            system, user = prompts.build_prompt(rec.query, docs)
            response = client.generate(system, user)
        refusals.append(matching.is_refusal(response))

    return [{
        "ability": "negative_rejection",
        "model": client.name,
        "mitigation": mitigation,
        "noise_rate": config.NEGATIVE_REJECTION_NOISE_RATE,
        "n": len(refusals),
        "rejection_rate": metrics.rejection_rate(refusals),
    }]
