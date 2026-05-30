"""
Convert a model1 results JSONL to Kaggle submission CSV.

Usage
-----
  python model1_to_submission.py --variant v2_fewshot
  python model1_to_submission.py --input results/model1_v2_fewshot_results.jsonl --output results/model1_submission.csv
"""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default=None,
                        help="Variant name, e.g. v2_fewshot (expands to results/model1_<variant>_results.jsonl)")
    parser.add_argument("--input", default=None, help="Explicit input JSONL path")
    parser.add_argument("--output", default=None, help="Output CSV path (default: results/model1_<variant>_submission.csv)")
    args = parser.parse_args()

    if args.input:
        in_path = Path(args.input)
    elif args.variant:
        in_path = Path(f"results/model1_{args.variant}_results.jsonl")
    else:
        parser.error("Provide --variant or --input")

    if args.output:
        out_path = Path(args.output)
    else:
        stem = in_path.stem.replace("_results", "")
        out_path = Path("results") / f"{stem}_submission.csv"

    records = [json.loads(l) for l in open(in_path)]
    records.sort(key=lambda r: r["id"])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "response"])
        for r in records:
            writer.writerow([r["id"], r["response"]])

    print(f"Wrote {len(records)} rows to {out_path}")


if __name__ == "__main__":
    main()
