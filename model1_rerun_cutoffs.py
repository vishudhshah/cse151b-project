"""
Two-pass fix for model1 results: rerun questions that were cut off mid-think
(i.e. no </think> in response) with a short thinking budget so the model is
forced to produce a complete answer rather than nothing.

Usage
-----
  python model1_rerun_cutoffs.py --results results/model1_v2_fewshot_private_results.jsonl
                                 --data data/private.jsonl
                                 --variant v2_fewshot

The script patches the results file in-place: cutoff records are replaced with
the new short-budget responses. Original non-cutoff records are untouched.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

# Short budget forces the model to wrap up quickly rather than think forever.
SHORT_THINKING_BUDGET = 512
SHORT_MAX_TOKENS = 2048

_SAMPLING_KWARGS = dict(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    repetition_penalty=1.0,
)

_CHUNK_SIZE = 50


def is_cutoff(response: str) -> bool:
    return "</think>" not in response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="Path to existing model1 results JSONL to patch")
    parser.add_argument("--data", required=True,
                        help="Path to original JSONL dataset (e.g. data/private.jsonl)")
    parser.add_argument("--variant", required=True,
                        help="Variant name used to build prompts (e.g. v2_fewshot)")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    results_path = Path(args.results)
    records = [json.loads(l) for l in open(results_path)]
    cutoff_ids = {r["id"] for r in records if is_cutoff(r["response"])}

    print(f"Total records  : {len(records)}")
    print(f"Cutoffs to fix : {len(cutoff_ids)}")

    if not cutoff_ids:
        print("No cutoffs found — nothing to do.")
        return

    # Load only the cutoff questions from the data file
    all_data = {json.loads(l)["id"]: json.loads(l) for l in open(args.data)}
    cutoff_items = [all_data[i] for i in sorted(cutoff_ids) if i in all_data]

    # Import variant config from model1
    sys.path.insert(0, str(Path(__file__).parent))
    from model1_prompt_engineering import VARIANTS, build_prompt
    from judger import Judger

    cfg = VARIANTS[args.variant]
    judger = Judger(strict_extract=False)

    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print(f"Loading model with vLLM...")
    llm = LLM(
        model=MODEL_ID,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    print("Model loaded.\n")

    sampling_params = SamplingParams(
        max_tokens=SHORT_MAX_TOKENS,
        **_SAMPLING_KWARGS,
    )

    # Run inference in chunks
    new_responses: dict[int, str] = {}
    for chunk_start in tqdm(range(0, len(cutoff_items), _CHUNK_SIZE), desc="Rerunning cutoffs"):
        chunk = cutoff_items[chunk_start: chunk_start + _CHUNK_SIZE]
        prompts = []
        for item in chunk:
            sys_p, usr_p = build_prompt(item["question"], item.get("options"), cfg)
            prompts.append(
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": sys_p},
                     {"role": "user",   "content": usr_p}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                    thinking_budget=SHORT_THINKING_BUDGET,
                )
            )
        outputs = llm.generate(prompts, sampling_params)
        for item, out in zip(chunk, outputs):
            new_responses[item["id"]] = out.outputs[0].text.strip()

    # Score new responses (private set has no answers, so correct=None)
    def score(item, response):
        if "answer" not in item:
            return None
        from model1_prompt_engineering import score_response
        return score_response(item, response, judger)

    # Patch records
    still_cutoff = 0
    patched = 0
    records_by_id = {r["id"]: r for r in records}
    for qid, response in new_responses.items():
        item = all_data[qid]
        records_by_id[qid] = {
            **records_by_id[qid],
            "response": response,
            "correct":  score(item, response),
            "rerun":    True,
        }
        if is_cutoff(response):
            still_cutoff += 1
        else:
            patched += 1

    print(f"\nPatched (now complete): {patched}")
    print(f"Still cut off         : {still_cutoff}")

    # Write patched results back
    with open(results_path, "w") as f:
        for r in sorted(records_by_id.values(), key=lambda x: x["id"]):
            f.write(json.dumps(r) + "\n")

    print(f"Saved patched results to {results_path}")


if __name__ == "__main__":
    main()
