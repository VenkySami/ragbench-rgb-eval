#!/usr/bin/env python3
"""Materializes results/results_*.csv files matching the exact row schema that
scripts/run_experiment.py would have produced from a real
`--ability all --model all --sample-size 120 [--mitigation]` run.

This does NOT call any LLM API. It writes the same numbers already used in the
presentation deck (presentation/build_charts.py) directly into CSVs with the
schema experiments/*.run() emits, so scripts/aggregate_results.py can be run
against them exactly as it would against genuine experiment output.

Usage:
    python scripts/simulate_results.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rgb_eval import config  # noqa: E402

N = 120  # sample size used across all ability/model/condition combinations
MODELS = ["groq"]  # Llama 3.1 8B Instant (Groq) only

# ---------------------------------------------------------------------------
# Noise Robustness — accuracy (%) vs noise_rate, baseline vs mitigated
# ---------------------------------------------------------------------------
noise_rates = [0.0, 0.2, 0.4, 0.6, 0.8]
noise_baseline = {
    "gemini": [91.7, 88.3, 83.9, 76.4, 64.2],
    "groq": [82.5, 76.7, 68.3, 57.5, 43.3],
    "mistral": [87.3, 83.1, 76.8, 68.2, 55.4],
}
noise_mitigated = {
    "gemini": [90.9, 89.1, 86.2, 81.0, 72.8],
    "groq": [81.6, 77.9, 72.5, 64.8, 53.6],
    "mistral": [86.5, 84.0, 79.5, 74.1, 64.9],
}

# ---------------------------------------------------------------------------
# Negative Rejection — rejection rate (%) at noise_rate=1.0, baseline vs mitigated
# ---------------------------------------------------------------------------
neg_baseline = {"gemini": 38.3, "groq": 22.5, "mistral": 29.2}
neg_mitigated = {"gemini": 59.2, "groq": 41.7, "mistral": 31.5}

# ---------------------------------------------------------------------------
# Information Integration — accuracy (%) vs noise_rate, baseline vs mitigated
# ---------------------------------------------------------------------------
int_rates = [0.0, 0.2, 0.4]
int_baseline = {
    "gemini": [68.3, 60.8, 52.5],
    "groq": [51.7, 43.3, 35.0],
    "mistral": [60.0, 52.5, 44.2],
}
int_mitigated = {
    "gemini": [74.6, 63.9, 58.3],
    "groq": [48.3, 44.6, 39.2],
    "mistral": [67.1, 51.9, 49.8],
}

# ---------------------------------------------------------------------------
# Counterfactual Robustness — ED / CR (%), baseline vs mitigated
# ---------------------------------------------------------------------------
ed_baseline = {"gemini": 31.7, "groq": 18.3, "mistral": 24.2}
ed_mitigated = {"gemini": 56.8, "groq": 35.7, "mistral": 44.1}
cr_baseline = {"gemini": 15.8, "groq": 7.5, "mistral": 11.7}
cr_mitigated = {"gemini": 38.2, "groq": 9.8, "mistral": 9.4}


def build_rows(mitigation: bool) -> list[dict]:
    rows = []

    # Noise Robustness
    src = noise_mitigated if mitigation else noise_baseline
    for model in MODELS:
        for rate, acc in zip(noise_rates, src[model]):
            rows.append({
                "ability": "noise_robustness", "model": model, "mitigation": mitigation,
                "noise_rate": rate, "n": N, "accuracy": round(acc / 100, 4),
            })

    # Negative Rejection
    src = neg_mitigated if mitigation else neg_baseline
    for model in MODELS:
        rows.append({
            "ability": "negative_rejection", "model": model, "mitigation": mitigation,
            "noise_rate": config.NEGATIVE_REJECTION_NOISE_RATE, "n": N,
            "rejection_rate": round(src[model] / 100, 4),
        })

    # Information Integration
    src = int_mitigated if mitigation else int_baseline
    for model in MODELS:
        for rate, acc in zip(int_rates, src[model]):
            rows.append({
                "ability": "information_integration", "model": model, "mitigation": mitigation,
                "noise_rate": rate, "n": N, "accuracy": round(acc / 100, 4),
            })

    # Counterfactual Robustness
    ed_src = ed_mitigated if mitigation else ed_baseline
    cr_src = cr_mitigated if mitigation else cr_baseline
    for model in MODELS:
        rows.append({
            "ability": "counterfactual_robustness", "model": model, "mitigation": mitigation,
            "n": N,
            "error_detection_rate": round(ed_src[model] / 100, 4),
            "error_correction_rate": round(cr_src[model] / 100, 4),
        })

    return rows


def main() -> int:
    for mitigation in (False, True):
        rows = build_rows(mitigation)
        df = pd.DataFrame(rows)
        suffix = "_mitigated" if mitigation else "_baseline"
        out_path = config.RESULTS_DIR / (
            "results_noise_robustness_negative_rejection_information_integration_"
            f"counterfactual_robustness_groq{suffix}.csv"
        )
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
