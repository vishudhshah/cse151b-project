"""
Model 2: Sampling Parameters & Majority Voting for CSE 151B Math Competition

Self-consistency (majority voting) generates N independent samples per question
and picks the most common answer, reducing variance from stochastic decoding.
This script also sweeps temperature to measure its effect on single-sample accuracy.

Experiments
-----------
  temp_sweep   — one sample per question at T ∈ {0.0, 0.3, 0.5, 0.7, 0.9}
                 T=0.0 uses greedy decoding
  voting_n3    — 3 samples per question, T=0.7, majority vote
  voting_n5    — 5 samples per question, T=0.7, majority vote
  voting_n7    — 7 samples per question, T=0.7, majority vote

Voting logic
------------
  MCQ:       Extract letter from each sample's \\boxed{}. Take the modal letter.
  Free-form: Extract \\boxed{} content from each sample; normalize with judger;
             take the modal normalized string. Wrap in \\boxed{} for final scoring.
  Tie-break: First sample's answer is used when no majority exists.

Usage
-----
  python model2_sampling_voting.py [options]

  --experiment {temp_sweep,voting_n3,voting_n5,voting_n7,all}   Default: all
  --limit N        Only run on first N questions
  --gpu ID         CUDA_VISIBLE_DEVICES (default: 0)
  --data PATH      Path to JSONL dataset (default: data/public.jsonl)
  --prompt VARIANT Prompt variant from Model 1 to use (default: v1_enhanced_cot)

Output
------
  results/model2_{experiment}_results.jsonl  — per-question records
  Printed summary table

Runtime estimate (A30 24 GB, vLLM, bfloat16, full 1126 questions)
------------------------------------------------------------------
  temp_sweep : ~15 min  (5 temperatures)
  voting_n5  : ~25 min  (5 samples generated in one vLLM call per chunk)
  voting_n7  : ~35 min  (7 samples generated in one vLLM call per chunk)
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_ID   = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH  = "data/public.jsonl"
MAX_TOKENS = 4096
THINKING_BUDGET = 3072

VOTING_TEMPERATURE = 0.7
TEMP_SWEEP_VALUES  = [0.0, 0.3, 0.5, 0.7, 0.9]

_CHUNK_SIZE = 50  # flush results to disk every N questions

# ── Prompt Definitions (mirrors Model 1 v1_enhanced_cot as default) ─────────────

SYSTEM_PROMPTS = {
    "v0_baseline": {
        "math": (
            "You are an expert mathematician. Solve the problem step-by-step. "
            "Put your final answer inside \\boxed{}. "
            "If the problem has multiple sub-answers, separate them by commas inside "
            "a single \\boxed{}, e.g. \\boxed{3, 7}."
        ),
        "mcq": (
            "You are an expert mathematician. "
            "Read the problem and the answer choices below, then select the single best answer. "
            "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
        ),
    },
    "v1_enhanced_cot": {
        "math": (
            "You are an expert mathematician. Solve the problem using the following structure:\n"
            "1. Identify what is being asked and what information is given.\n"
            "2. Select the relevant formulas or theorems.\n"
            "3. Carry out each calculation step, showing your work.\n"
            "4. Verify your answer.\n"
            "5. Write your final answer inside \\boxed{}.\n"
            "If the problem has multiple [ANS] placeholders, separate all answers by commas "
            "inside a single \\boxed{}, e.g. \\boxed{3, 7}."
        ),
        "mcq": (
            "You are an expert mathematician.\n"
            "1. Read the question and understand what it asks.\n"
            "2. Compute or reason about the answer independently.\n"
            "3. Match your result to the best option.\n"
            "4. Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
        ),
    },
}

# ── Prompt Building ────────────────────────────────────────────────────────────

def build_prompt(question: str, options: Optional[list], prompt_variant: str) -> tuple[str, str]:
    cfg = SYSTEM_PROMPTS.get(prompt_variant, SYSTEM_PROMPTS["v1_enhanced_cot"])
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return cfg["mcq"], f"{question}\n\nOptions:\n{opts_text}"
    return cfg["math"], question

# ── Scoring Helpers ────────────────────────────────────────────────────────────

def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_single(item: dict, response: str, judger) -> bool:
    is_mcq = bool(item.get("options"))
    gold   = item["answer"]
    if is_mcq:
        return extract_letter(response) == str(gold).strip().upper()
    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        return judger.auto_judge(pred=response, gold=gold_list,
                                 options=[[]] * len(gold_list))
    except Exception:
        return False

# ── Majority Voting ────────────────────────────────────────────────────────────

def majority_vote_mcq(responses: list[str]) -> str:
    letters = [extract_letter(r) for r in responses]
    letters = [l for l in letters if l]
    if not letters:
        return ""
    return Counter(letters).most_common(1)[0][0]


def majority_vote_free(responses: list[str], judger) -> str:
    normalized = []
    for r in responses:
        raw = judger.extract_boxed_answer(r)
        if raw:
            normalized.append(judger.norm_ans_str(raw))

    if not normalized:
        return ""

    counts = Counter(normalized)
    modal, modal_count = counts.most_common(1)[0]

    if modal_count > len(responses) / 2:
        return modal
    return normalized[0]


def score_voted(item: dict, voted_answer: str, judger) -> bool:
    is_mcq = bool(item.get("options"))
    gold   = item["answer"]
    if is_mcq:
        return voted_answer.upper() == str(gold).strip().upper()
    gold_list = gold if isinstance(gold, list) else [gold]
    synthetic_response = f"\\boxed{{{voted_answer}}}"
    try:
        return judger.auto_judge(pred=synthetic_response, gold=gold_list,
                                 options=[[]] * len(gold_list))
    except Exception:
        return False


def agreement_rate(responses: list[str], voted: str, item: dict, judger) -> float:
    is_mcq = bool(item.get("options"))
    if is_mcq:
        letters = [extract_letter(r) for r in responses]
        agree   = sum(l == voted for l in letters if l)
        return agree / len(responses)
    normed = []
    for r in responses:
        raw = judger.extract_boxed_answer(r)
        if raw:
            normed.append(judger.norm_ans_str(raw))
    if not normed:
        return 0.0
    voted_norm = judger.norm_ans_str(voted) if voted else ""
    return sum(n == voted_norm for n in normed) / len(responses)

# ── Prompt Formatting ──────────────────────────────────────────────────────────

def format_prompts(tokenizer, items: list[dict], prompt_variant: str) -> list[str]:
    prompt_texts = []
    for item in items:
        sys_p, usr_p = build_prompt(item["question"], item.get("options"), prompt_variant)
        prompt_texts.append(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_p},
                 {"role": "user",   "content": usr_p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
                thinking_budget=THINKING_BUDGET,
            )
        )
    return prompt_texts

# ── Experiment Runners ─────────────────────────────────────────────────────────

def _load_done_ids(path: Path) -> set[int]:
    done = set()
    if path.exists():
        for line in open(path):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    return done


def _summary_row_from_file(path: Path, label: str) -> dict:
    results = [json.loads(l) for l in open(path)]
    mcq  = [r for r in results if r["is_mcq"]]
    free = [r for r in results if not r["is_mcq"]]
    def acc(s): return sum(r["correct"] for r in s) / len(s) * 100 if s else 0.0
    return {"label": label, "mcq_acc": acc(mcq), "free_acc": acc(free), "total_acc": acc(results)}


def run_temp_sweep(llm, tokenizer, data: list, prompt_variant: str,
                   judger, out_dir: Path, max_tokens: int) -> list[dict]:
    print(f"\n{'─'*60}")
    print(f"  Experiment : temp_sweep  Temperatures: {TEMP_SWEEP_VALUES}")
    print(f"{'─'*60}")

    summary_rows = []
    for temp in TEMP_SWEEP_VALUES:
        out_path  = out_dir / f"model2_temp{str(temp).replace('.', 'p')}_results.jsonl"
        done_ids  = _load_done_ids(out_path)
        remaining = [d for d in data if d["id"] not in done_ids]

        if done_ids:
            print(f"  T={temp}: resuming — {len(done_ids)} done, {len(remaining)} remaining")

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temp,
            top_p=0.95,
            top_k=20 if temp > 0 else -1,
            repetition_penalty=1.0,
        )

        with open(out_path, "a") as f_out:
            with tqdm(total=len(remaining), desc=f"  T={temp}") as pbar:
                for chunk_start in range(0, len(remaining), _CHUNK_SIZE):
                    chunk = remaining[chunk_start : chunk_start + _CHUNK_SIZE]
                    prompt_texts = format_prompts(tokenizer, chunk, prompt_variant)
                    outputs = llm.generate(prompt_texts, sampling_params)
                    for item, output in zip(chunk, outputs):
                        response = output.outputs[0].text.strip()
                        record = {
                            "id":          item["id"],
                            "experiment":  "temp_sweep",
                            "temperature": temp,
                            "n_samples":   1,
                            "is_mcq":      bool(item.get("options")),
                            "gold":        item["answer"],
                            "responses":   [response],
                            "voted":       response,
                            "agreement":   1.0,
                            "correct":     score_single(item, response, judger),
                        }
                        f_out.write(json.dumps(record) + "\n")
                    pbar.update(len(chunk))
                    f_out.flush()

        label = f"T={temp}" + (" (greedy)" if temp == 0.0 else "")
        row   = _summary_row_from_file(out_path, label)
        print(f"  {label}: MCQ={row['mcq_acc']:.2f}%  Free={row['free_acc']:.2f}%"
              f"  Overall={row['total_acc']:.2f}%  → {out_path}")
        summary_rows.append(row)

    return summary_rows


def run_voting(n_samples: int, llm, tokenizer, data: list, prompt_variant: str,
               judger, out_dir: Path, max_tokens: int) -> dict:
    exp_name = f"voting_n{n_samples}"
    out_path = out_dir / f"model2_{exp_name}_results.jsonl"
    done_ids = _load_done_ids(out_path)
    remaining = [d for d in data if d["id"] not in done_ids]

    print(f"\n{'─'*60}")
    print(f"  Experiment : {exp_name}  Samples/Q: {n_samples}  T: {VOTING_TEMPERATURE}")
    if done_ids:
        print(f"  Resuming: {len(done_ids)} done, {len(remaining)} remaining")
    print(f"{'─'*60}")

    # n=n_samples tells vLLM to produce N independent completions per prompt in one pass
    sampling_params = SamplingParams(
        n=n_samples,
        max_tokens=max_tokens,
        temperature=VOTING_TEMPERATURE,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.0,
    )

    with open(out_path, "a") as f_out:
        with tqdm(total=len(remaining), desc=f"  {exp_name}") as pbar:
            for chunk_start in range(0, len(remaining), _CHUNK_SIZE):
                chunk = remaining[chunk_start : chunk_start + _CHUNK_SIZE]
                prompt_texts = format_prompts(tokenizer, chunk, prompt_variant)
                outputs = llm.generate(prompt_texts, sampling_params)
                for item, output in zip(chunk, outputs):
                    responses = [o.text.strip() for o in output.outputs]
                    is_mcq    = bool(item.get("options"))
                    voted     = majority_vote_mcq(responses) if is_mcq else majority_vote_free(responses, judger)
                    agree     = agreement_rate(responses, voted, item, judger)
                    correct   = score_voted(item, voted, judger)
                    record    = {
                        "id":          item["id"],
                        "experiment":  exp_name,
                        "n_samples":   n_samples,
                        "temperature": VOTING_TEMPERATURE,
                        "is_mcq":      is_mcq,
                        "gold":        item["answer"],
                        "responses":   responses,
                        "voted":       voted,
                        "agreement":   agree,
                        "correct":     correct,
                    }
                    f_out.write(json.dumps(record) + "\n")
                pbar.update(len(chunk))
                f_out.flush()

    results   = [json.loads(l) for l in open(out_path)]
    mcq_res   = [r for r in results if r["is_mcq"]]
    free_res  = [r for r in results if not r["is_mcq"]]
    def acc(s): return sum(r["correct"] for r in s) / len(s) * 100 if s else 0.0
    avg_agree = sum(r["agreement"] for r in results) / len(results) * 100

    row = {
        "label":     f"voting_n{n_samples}",
        "n_samples": n_samples,
        "mcq_acc":   acc(mcq_res),
        "free_acc":  acc(free_res),
        "total_acc": acc(results),
        "avg_agree": avg_agree,
    }
    print(f"  MCQ={row['mcq_acc']:.2f}%  Free={row['free_acc']:.2f}%"
          f"  Overall={row['total_acc']:.2f}%  Avg-agreement={avg_agree:.1f}%")
    print(f"  Saved to : {out_path}")
    return row

# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Model 2: Sampling Parameters & Majority Voting — CSE 151B"
    )
    parser.add_argument("--experiment", default="all",
                        choices=["temp_sweep", "voting_n3", "voting_n5", "voting_n7", "all"],
                        help="Which experiment to run (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate on first N questions")
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS,
                        help=f"Max new tokens per response (default: {MAX_TOKENS}). "
                             "Use 2048 for fast smoke tests.")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES (default: 0)")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--prompt", default="v1_enhanced_cot",
                        choices=list(SYSTEM_PROMPTS),
                        help="Prompt variant from Model 1 (default: v1_enhanced_cot)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    data = [json.loads(line) for line in open(args.data)]
    if args.limit:
        data = data[: args.limit]
    n_mcq  = sum(bool(d.get("options")) for d in data)
    print(f"Dataset : {args.data}")
    print(f"Questions: {len(data)}  ({n_mcq} MCQ, {len(data)-n_mcq} free-form)")
    print(f"Prompt  : {args.prompt}")

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

    sys.path.insert(0, str(Path(__file__).parent))
    from judger import Judger
    judger = Judger(strict_extract=False)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    experiments = (
        ["temp_sweep", "voting_n3", "voting_n5", "voting_n7"]
        if args.experiment == "all"
        else [args.experiment]
    )

    temp_rows   = []
    voting_rows = []

    for exp in experiments:
        if exp == "temp_sweep":
            temp_rows = run_temp_sweep(llm, tokenizer, data, args.prompt, judger,
                                       out_dir, args.max_tokens)
        else:
            n = int(exp.replace("voting_n", ""))
            voting_rows.append(run_voting(n, llm, tokenizer, data, args.prompt, judger,
                                          out_dir, args.max_tokens))

    # Print final summary table
    W = 72
    print(f"\n{'='*W}")
    print(f"  SUMMARY — Model 2: Sampling & Voting  ({len(data)} questions)")
    print(f"{'='*W}")

    if temp_rows:
        print(f"  {'Experiment':<22} {'MCQ':>8} {'Free-form':>12} {'Overall':>10}")
        print(f"  {'-'*(W-4)}")
        for row in temp_rows:
            print(f"  {row['label']:<22} {row['mcq_acc']:>7.2f}%"
                  f" {row['free_acc']:>11.2f}% {row['total_acc']:>9.2f}%")

    if voting_rows:
        print(f"\n  {'Experiment':<22} {'MCQ':>8} {'Free-form':>12} {'Overall':>10} {'Avg-agree':>12}")
        print(f"  {'-'*(W-4)}")
        for row in voting_rows:
            print(f"  {row['label']:<22} {row['mcq_acc']:>7.2f}%"
                  f" {row['free_acc']:>11.2f}% {row['total_acc']:>9.2f}%"
                  f" {row['avg_agree']:>11.1f}%")

    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
