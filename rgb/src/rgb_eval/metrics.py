"""Hand-written metric formulas (per task instructions: write the counting math
ourselves, do not reuse the reference repo's implementation).

All functions take lists of booleans/results already computed by an experiment
runner and return a single float in [0, 1].
"""
from __future__ import annotations


def accuracy(results: list[bool]) -> float:
    """Fraction of examples whose response contained a gold answer alias.
    Used for Noise Robustness and Information Integration."""
    if not results:
        return 0.0
    return sum(1 for r in results if r) / len(results)


def rejection_rate(refusals: list[bool]) -> float:
    """Fraction of negative-rejection-testbed responses that issued the exact
    refusal phrase (Rej in the paper; Rej* LLM-judged variant is excluded)."""
    if not refusals:
        return 0.0
    return sum(1 for r in refusals if r) / len(refusals)


def error_detection_rate(flags: list[bool]) -> float:
    """Fraction of counterfactual-document responses that flagged a factual
    error (ED in the paper; ED* LLM-judged variant is excluded)."""
    if not flags:
        return 0.0
    return sum(1 for f in flags if f) / len(flags)


def error_correction_rate(flags: list[bool], corrected: list[bool]) -> float:
    """Of the examples where an error WAS detected, fraction where the response
    also gave the correct (non-fake) answer (CR in the paper)."""
    detected_idx = [i for i, f in enumerate(flags) if f]
    if not detected_idx:
        return 0.0
    return sum(1 for i in detected_idx if corrected[i]) / len(detected_idx)
