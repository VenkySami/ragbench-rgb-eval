"""Information Integration experiment: accuracy across noise_rate in
{0, 0.2, 0.4} on en_int.json, with an optional mitigation (query decomposition
+ merge).
"""
from __future__ import annotations

from .. import config, data_loader, matching, metrics, prompts
from ..llm_clients import BaseLLMClient
from ..mitigations.decompose_integrate import decompose_and_answer


def run(client: BaseLLMClient, sample_size: int | None = None, mitigation: bool = False) -> list[dict]:
    records = data_loader.load_records("information_integration")
    records = data_loader.sample_records(records, sample_size)

    rows = []
    for noise_rate in config.INTEGRATION_NOISE_RATES:
        results = []
        for rec in records:
            docs = data_loader.sample_docs(rec, noise_rate, config.PASSAGE_NUM)
            if mitigation:
                response = decompose_and_answer(client, rec.query, docs)
            else:
                system, user = prompts.build_prompt(rec.query, docs)
                response = client.generate(system, user)
            results.append(matching.contains_answer(response, rec.answer))

        rows.append({
            "ability": "information_integration",
            "model": client.name,
            "mitigation": mitigation,
            "noise_rate": noise_rate,
            "n": len(results),
            "accuracy": metrics.accuracy(results),
        })
    return rows
