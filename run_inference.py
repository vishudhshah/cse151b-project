"""
run_inference.py — Final submission entry point for CSE 151B competition.

Strategy: Model 1 v2_fewshot — few-shot examples + enhanced CoT on
Qwen/Qwen3-4B-Thinking-2507 (base model, no fine-tuning).

Usage
-----
  # Python API (default: runs on data/private.jsonl)
  from run_inference import run_inference
  run_inference()

  # CLI — private set (Kaggle submission)
  python run_inference.py

  # CLI — public set (accuracy verification)
  python run_inference.py --data data/public.jsonl
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Reuse config and utilities from model1 to keep hyperparameters in sync
from model1_prompt_engineering import (
    MODEL_ID,
    MAX_TOKENS,
    THINKING_BUDGET,
    _SAMPLING_KWARGS,
    VARIANTS,
    _format_prompts,
    score_response,
)

_VARIANT = "v2_fewshot"
_CHUNK_SIZE = 50


def run_inference(
    data_path: str = "data/private.jsonl",
    output_csv: str = "results/model1_submission.csv",
    gpu: str = "0",
) -> Path:
    """Run Model 1 v2_fewshot end-to-end and write the submission CSV.

    Parameters
    ----------
    data_path : path to input JSONL (public.jsonl or private.jsonl)
    output_csv: path for the output submission CSV
    gpu       : CUDA_VISIBLE_DEVICES value

    Returns
    -------
    Path to the written CSV file.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    data = [json.loads(line) for line in open(data_path)]
    is_private = not any("answer" in d for d in data)
    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = len(data) - n_mcq
    print(f"Dataset  : {data_path}")
    print(f"Questions: {len(data)}  ({n_mcq} MCQ, {n_free} free-form)")
    print(f"Mode     : {'private (no scoring)' if is_private else 'public (with accuracy)'}")

    print(f"\nLoading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print(f"Loading {MODEL_ID} with vLLM (bfloat16)...")
    llm = LLM(
        model=MODEL_ID,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    print("Model loaded.\n")

    sampling_params = SamplingParams(max_tokens=MAX_TOKENS, **_SAMPLING_KWARGS)
    cfg = VARIANTS[_VARIANT]

    sys.path.insert(0, str(Path(__file__).parent))
    from judger import Judger
    judger = Judger(strict_extract=False)

    results = []
    for chunk_start in range(0, len(data), _CHUNK_SIZE):
        chunk = data[chunk_start : chunk_start + _CHUNK_SIZE]
        prompt_texts = _format_prompts(tokenizer, chunk, cfg)
        outputs = llm.generate(prompt_texts, sampling_params)
        for item, output in zip(chunk, outputs):
            response = output.outputs[0].text.strip()
            record: dict = {"id": item["id"], "response": response}
            if not is_private:
                record["correct"] = score_response(item, response, judger)
            results.append(record)
        print(f"  {min(chunk_start + _CHUNK_SIZE, len(data))}/{len(data)} done")

    results.sort(key=lambda r: r["id"])

    out_path = Path(output_csv)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "response"])
        for r in results:
            writer.writerow([r["id"], r["response"]])
    print(f"\nSubmission CSV: {out_path}  ({len(results)} rows)")

    if not is_private:
        correct = sum(r["correct"] for r in results)
        mcq_res  = [r for r in results if any(
            d["id"] == r["id"] and bool(d.get("options")) for d in data)]
        total = len(results)
        print(f"Overall accuracy: {correct}/{total} ({correct/total*100:.2f}%)")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model 1 v2_fewshot inference — CSE 151B final submission"
    )
    parser.add_argument("--data",   default="data/private.jsonl",
                        help="Input JSONL dataset (default: data/private.jsonl)")
    parser.add_argument("--output", default="results/model1_submission.csv",
                        help="Output CSV path (default: results/model1_submission.csv)")
    parser.add_argument("--gpu",    default="0",
                        help="CUDA_VISIBLE_DEVICES (default: 0)")
    args = parser.parse_args()
    run_inference(data_path=args.data, output_csv=args.output, gpu=args.gpu)
