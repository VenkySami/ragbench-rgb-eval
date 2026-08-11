#!/usr/bin/env python3
"""Aggregates every results/*.csv produced by run_experiment.py into the 4
paper-style summary tables (baseline and, if present, mitigated) and prints /
saves them for the presentation.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rgb_eval import config  # noqa: E402


def load_all() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in config.RESULTS_DIR.glob("results_*.csv")]
    if not frames:
        print("No results_*.csv files found in results/. Run scripts/run_experiment.py first.")
        sys.exit(1)
    return pd.concat(frames, ignore_index=True)


def table_noise_robustness(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["ability"] == "noise_robustness"]
    return sub.pivot_table(index="noise_rate", columns=["model", "mitigation"], values="accuracy")


def table_negative_rejection(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["ability"] == "negative_rejection"]
    return sub.pivot_table(index="mitigation", columns="model", values="rejection_rate")


def table_information_integration(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["ability"] == "information_integration"]
    return sub.pivot_table(index="noise_rate", columns=["model", "mitigation"], values="accuracy")


def table_counterfactual(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["ability"] == "counterfactual_robustness"]
    return sub.pivot_table(
        index="mitigation", columns="model", values=["error_detection_rate", "error_correction_rate"]
    )


def main() -> int:
    df = load_all()
    tables = {
        "noise_robustness": table_noise_robustness(df),
        "negative_rejection": table_negative_rejection(df),
        "information_integration": table_information_integration(df),
        "counterfactual_robustness": table_counterfactual(df),
    }
    for name, table in tables.items():
        print(f"\n=== {name} ===")
        print(table)
        out_path = config.RESULTS_DIR / f"summary_{name}.csv"
        table.to_csv(out_path)
        print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
