"""Run a rank-ratio sweep and print (and optionally save) the results.

    python -m src.run_experiment --rank-ratio 1.0 0.7 0.5 --out results/sweep.csv

A rank ratio of 1.0 is the baseline: the model as shipped, not factorized.
"""

import argparse
import csv
from pathlib import Path

import torch

from src.compress import block_params, compress_model
from src.eval import load_eval_ids, load_model, load_tokenizer, perplexity
from src.latency import median_latency_ms

COLUMNS = ["rank_ratio", "block_params", "reduction_pct", "perplexity", "delta_pct_vs_baseline"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rank-ratio", type=float, nargs="+", default=[1.0])
    parser.add_argument("--out", type=Path, help="write the results table as CSV")
    parser.add_argument("--latency", action="store_true", help="also measure CPU latency (slower)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer()
    input_ids = load_eval_ids(tokenizer)
    model = load_model()
    base_params = block_params(model)

    rows = []
    for ratio in args.rank_ratio:
        candidate = model if ratio >= 1.0 else compress_model(model, ratio)
        params = block_params(candidate)
        torch.manual_seed(args.seed)  # same seed for every measurement
        row = {
            "rank_ratio": ratio,
            "block_params": params,
            "reduction_pct": round(100 * (1 - params / base_params), 2),
            "perplexity": round(perplexity(candidate, input_ids), 4),
        }
        if args.latency:
            row["latency_ms"] = round(median_latency_ms(candidate, input_ids), 2)
        rows.append(row)

    baseline = next((r for r in rows if r["rank_ratio"] >= 1.0), None)
    for row in rows:
        row["delta_pct_vs_baseline"] = (
            round(100 * (row["perplexity"] / baseline["perplexity"] - 1), 2) if baseline else None
        )

    columns = COLUMNS + (["latency_ms"] if args.latency else [])
    print("\t".join(columns))
    for row in rows:
        print("\t".join(str(row.get(c, "")) for c in columns))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
