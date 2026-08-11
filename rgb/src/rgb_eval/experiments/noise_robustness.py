"""Noise Robustness experiment: accuracy across noise_rate in {0, 0.2, 0.4, 0.6, 0.8}
on en_refine.json, with an optional mitigation (relevance filtering) applied.
"""
from __future__ import annotations

from .. import config, data_loader, matching, metrics, prompts
from ..llm_clients import BaseLLMClient
from ..mitigations.noise_filter import filter_relevant_docs


def run(client: BaseLLMClient, sample_size: int | None = None, mitigation: bool = False) -> list[dict]:
    records = data_loader.load_records("noise_robustness")
    records = data_loader.sample_records(records, sample_size)

    rows = []
    for noise_rate in config.NOISE_RATES:
        results = []
        for rec in records:
            docs = data_loader.sample_docs(rec, noise_rate, config.PASSAGE_NUM)
            if mitigation:
                docs = filter_relevant_docs(rec.query, docs)
            system, user = prompts.build_prompt(rec.query, docs)
            response = client.generate(system, user)
            results.append(matching.contains_answer(response, rec.answer))

        rows.append({
            "ability": "noise_robustness",
            "model": client.name,
            "mitigation": mitigation,
            "noise_rate": noise_rate,
            "n": len(results),
            "accuracy": metrics.accuracy(results),
        })
    return rows
