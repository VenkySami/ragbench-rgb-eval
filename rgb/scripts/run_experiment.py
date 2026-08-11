#!/usr/bin/env python3
"""CLI runner for the RGB capstone experiments.

Examples:
    python scripts/run_experiment.py --ability noise_robustness --model gemini --sample-size 60
    python scripts/run_experiment.py --ability all --model all --sample-size 60 --mitigation
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rgb_eval import config  # noqa: E402
from rgb_eval.llm_clients import get_client  # noqa: E402
from rgb_eval.experiments import (  # noqa: E402
    noise_robustness,
    negative_rejection,
    information_integration,
    counterfactual_robustness,
)

ABILITY_RUNNERS = {
    "noise_robustness": noise_robustness.run,
    "negative_rejection": negative_rejection.run,
    "information_integration": information_integration.run,
    "counterfactual_robustness": counterfactual_robustness.run,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ability", choices=list(ABILITY_RUNNERS) + ["all"], required=True)
    parser.add_argument("--model", choices=list(config.MODELS) + ["all"], required=True)
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--mitigation", action="store_true")
    args = parser.parse_args()

    abilities = list(ABILITY_RUNNERS) if args.ability == "all" else [args.ability]
    models = list(config.MODELS) if args.model == "all" else [args.model]

    all_rows = []
    for model_name in models:
        client = get_client(model_name)
        for ability in abilities:
            print(f"--- Running {ability} on {model_name} (mitigation={args.mitigation}) ---")
            rows = ABILITY_RUNNERS[ability](client, sample_size=args.sample_size, mitigation=args.mitigation)
            all_rows.extend(rows)
            for row in rows:
                print(row)

    suffix = "_mitigated" if args.mitigation else "_baseline"
    out_path = config.RESULTS_DIR / f"results_{'_'.join(abilities)}_{'_'.join(models)}{suffix}.csv"
    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
