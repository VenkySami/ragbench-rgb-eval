"""Loads RGB English json files and builds noise-controlled document contexts.

Own sampling implementation (not copied from the reference repo): for a given
noise_rate and passage_num, we deterministically sample
    num_positive = round(passage_num * (1 - noise_rate))
    num_negative = passage_num - num_positive
documents from the record's own `positive`/`negative` pools, using a fixed seed
per (record id, noise_rate) so results are reproducible and cache-friendly.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class Record:
    id: int
    query: str
    answer: object          # nested list of alias lists (accuracy datasets) or str (fact dataset)
    positive: list
    negative: list
    extra: dict = field(default_factory=dict)  # dataset-specific fields (fakeanswer, answer1/2, ...)


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_records(ability: str) -> list[Record]:
    path = config.DATA_FILES[ability]
    raw = _load_jsonl(path)
    records = []
    for r in raw:
        known = {"id", "query", "answer", "positive", "negative"}
        extra = {k: v for k, v in r.items() if k not in known}
        records.append(
            Record(
                id=r["id"],
                query=r["query"],
                answer=r.get("answer"),
                positive=r.get("positive") or [],
                negative=r.get("negative") or [],
                extra=extra,
            )
        )
    return records


def _flatten(docs: list) -> list[str]:
    """en_int.json's `positive` is a list-of-lists (one list of snippets per
    sub-question); flatten to a single list of passage strings."""
    flat = []
    for d in docs:
        if isinstance(d, list):
            flat.extend(d)
        else:
            flat.append(d)
    return flat


def sample_docs(record: Record, noise_rate: float, passage_num: int = config.PASSAGE_NUM) -> list[str]:
    """Build the passage_num-document context for a record at a given noise_rate."""
    rng = random.Random(f"{record.id}-{noise_rate}-{config.RANDOM_SEED}")
    positive = _flatten(record.positive)
    negative = _flatten(record.negative)

    num_positive = round(passage_num * (1 - noise_rate))
    num_negative = passage_num - num_positive

    # Clamp to what's available; pad the shortfall from the other pool so we
    # always return passage_num docs when the combined pool is large enough.
    num_positive = min(num_positive, len(positive))
    num_negative = min(num_negative, len(negative))
    shortfall = passage_num - (num_positive + num_negative)
    if shortfall > 0 and len(negative) > num_negative:
        extra = min(shortfall, len(negative) - num_negative)
        num_negative += extra
        shortfall -= extra
    if shortfall > 0 and len(positive) > num_positive:
        extra = min(shortfall, len(positive) - num_positive)
        num_positive += extra

    chosen_positive = rng.sample(positive, num_positive) if num_positive else []
    chosen_negative = rng.sample(negative, num_negative) if num_negative else []
    docs = chosen_positive + chosen_negative
    rng.shuffle(docs)
    return docs


def sample_records(records: list[Record], sample_size: int | None) -> list[Record]:
    if sample_size is None or sample_size >= len(records):
        return records
    rng = random.Random(config.RANDOM_SEED)
    return rng.sample(records, sample_size)
