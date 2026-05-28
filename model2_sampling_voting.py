"""
Model 2: Sampling Parameters & Majority Voting for CSE 151B Math Competition

Self-consistency (majority voting) generates N independent samples per question
and picks the most common answer, reducing variance from stochastic decoding.
This script also sweeps temperature to measure its effect on single-sample accuracy.

Experiments
-----------
  temp_sweep   — one sample per question at T ∈ {0.0, 0.3, 0.5, 0.7, 0.9}
                 T=0.0 uses greedy decoding (do_sample=False)
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

Runtime estimate (A100 40 GB, full 1126 questions)
---------------------------------------------------
  temp_sweep : ~60 min  (5 temperatures × ~12 min)
  voting_n5  : ~60 min  (5 samples × ~12 min)
  voting_n7  : ~84 min  (7 samples × ~12 min)
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_ID   = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH  = "data/public.jsonl"
MAX_TOKENS = 16384
THINKING_BUDGET = 3072

VOTING_TEMPERATURE = 0.7   # temperature used for all majority-voting experiments
TEMP_SWEEP_VALUES  = [0.0, 0.3, 0.5, 0.7, 0.9]

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
    """Return the modal letter extracted from a list of responses."""
    letters = [extract_letter(r) for r in responses]
    letters = [l for l in letters if l]  # drop blanks
    if not letters:
        return ""
    return Counter(letters).most_common(1)[0][0]


def majority_vote_free(responses: list[str], judger) -> str:
    """
    Extract and normalize \\boxed{} answers from each response, return the most
    common normalized string. Falls back to first extractable answer on ties.
    """
    normalized = []
    for r in responses:
        raw = judger.extract_boxed_answer(r)
        if raw:
            normalized.append(judger.norm_ans_str(raw))

    if not normalized:
        return ""

    counts = Counter(normalized)
    modal, modal_count = counts.most_common(1)[0]

    # If there's a strict majority (>50%) use it; otherwise fall back to first sample.
    if modal_count > len(responses) / 2:
        return modal
    return normalized[0]


def score_voted(item: dict, voted_answer: str, judger) -> bool:
    """Score a voted answer (string) against the ground truth."""
    is_mcq = bool(item.get("options"))
    gold   = item["answer"]
    if is_mcq:
        return voted_answer.upper() == str(gold).strip().upper()
    gold_list = gold if isinstance(gold, list) else [gold]
    # Wrap in \\boxed{} so auto_judge's extraction finds it
    synthetic_response = f"\\boxed{{{voted_answer}}}"
    try:
        return judger.auto_judge(pred=synthetic_response, gold=gold_list,
                                 options=[[]] * len(gold_list))
    except Exception:
        return False


def agreement_rate(responses: list[str], voted: str, item: dict, judger) -> float:
    """Fraction of samples that agree with the voted answer."""
    is_mcq = bool(item.get("options"))
    if is_mcq:
        letters = [extract_letter(r) for r in responses]
        agree   = sum(l == voted for l in letters if l)
        return agree / len(responses)
    normed  = []
    for r in responses:
        raw = judger.extract_boxed_answer(r)
        if raw:
            normed.append(judger.norm_ans_str(raw))
    if not normed:
        return 0.0
    voted_norm = judger.norm_ans_str(voted) if voted else ""
    return sum(n == voted_norm for n in normed) / len(responses)

# ── Single-Sample Inference ────────────────────────────────────────────────────

def generate_batch(llm, tokenizer, items: list[dict], prompt_variant: str,
                   temperature: float, do_sample: bool) -> list[str]:
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
    inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192,
    ).to(llm.device)

    gen_kwargs = dict(
        max_new_tokens=MAX_TOKENS,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs["do_sample"] = False  # greedy

    with torch.no_grad():
        output_ids = llm.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    responses = []
    for i in range(len(items)):
        new_tokens = output_ids[i][input_len:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return responses

# ── Experiment Runners ─────────────────────────────────────────────────────────

def _load_done_ids(path: Path) -> set[int]:
    """Return set of question IDs already written to a result file."""
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
                   judger, out_dir: Path, batch_size: int) -> list[dict]:
    """Single-sample at each temperature; resumes from partial output if present."""
    print(f"\n{'─'*60}")
    print(f"  Experiment : temp_sweep  Temperatures: {TEMP_SWEEP_VALUES}")
    print(f"{'─'*60}")

    summary_rows = []
    for temp in TEMP_SWEEP_VALUES:
        do_sample = temp > 0.0
        out_path  = out_dir / f"model2_temp{str(temp).replace('.', 'p')}_results.jsonl"
        done_ids  = _load_done_ids(out_path)
        remaining = [d for d in data if d["id"] not in done_ids]

        if done_ids:
            print(f"  T={temp}: resuming — {len(done_ids)} done, {len(remaining)} remaining")

        with open(out_path, "a") as f_out:
            with tqdm(total=len(remaining), desc=f"  T={temp}") as pbar:
                for batch_start in range(0, len(remaining), batch_size):
                    batch = remaining[batch_start : batch_start + batch_size]
                    responses = generate_batch(llm, tokenizer, batch, prompt_variant, temp, do_sample)
                    for item, response in zip(batch, responses):
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
                    pbar.update(len(batch))
                    f_out.flush()

        label = f"T={temp}" + (" (greedy)" if not do_sample else "")
        row   = _summary_row_from_file(out_path, label)
        print(f"  {label}: MCQ={row['mcq_acc']:.2f}%  Free={row['free_acc']:.2f}%"
              f"  Overall={row['total_acc']:.2f}%  → {out_path}")
        summary_rows.append(row)

    return summary_rows


def run_voting(n_samples: int, llm, tokenizer, data: list, prompt_variant: str,
               judger, out_dir: Path, batch_size: int) -> dict:
    """N-sample majority voting; resumes from partial output if present."""
    exp_name = f"voting_n{n_samples}"
    out_path = out_dir / f"model2_{exp_name}_results.jsonl"
    done_ids = _load_done_ids(out_path)
    remaining = [d for d in data if d["id"] not in done_ids]

    print(f"\n{'─'*60}")
    print(f"  Experiment : {exp_name}  Samples/Q: {n_samples}  T: {VOTING_TEMPERATURE}")
    if done_ids:
        print(f"  Resuming: {len(done_ids)} done, {len(remaining)} remaining")
    print(f"{'─'*60}")

    with open(out_path, "a") as f_out:
        with tqdm(total=len(remaining), desc=f"  {exp_name}") as pbar:
            for batch_start in range(0, len(remaining), batch_size):
                batch = remaining[batch_start : batch_start + batch_size]
                # n_samples rounds of batched generation; all_responses[i] holds n_samples strings for batch[i]
                all_responses = [[] for _ in batch]
                for _ in range(n_samples):
                    round_resps = generate_batch(llm, tokenizer, batch, prompt_variant,
                                                 VOTING_TEMPERATURE, True)
                    for i, r in enumerate(round_resps):
                        all_responses[i].append(r)
                for item, responses in zip(batch, all_responses):
                    is_mcq  = bool(item.get("options"))
                    voted   = majority_vote_mcq(responses) if is_mcq else majority_vote_free(responses, judger)
                    agree   = agreement_rate(responses, voted, item, judger)
                    correct = score_voted(item, voted, judger)
                    record  = {
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
                pbar.update(len(batch))
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
    global MAX_TOKENS
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
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Number of prompts per GPU call (default: 2, tuned for A30 24 GB). "
                             "Reduce to 1 if you get CUDA OOM errors; increase to 4 on A100 40 GB.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    MAX_TOKENS = args.max_tokens

    data = [json.loads(line) for line in open(args.data)]
    if args.limit:
        data = data[: args.limit]
    n_mcq  = sum(bool(d.get("options")) for d in data)
    print(f"Dataset : {args.data}")
    print(f"Questions: {len(data)}  ({n_mcq} MCQ, {len(data)-n_mcq} free-form)")
    print(f"Prompt  : {args.prompt}")
    print(f"Batch size: {args.batch_size}")

    print(f"\nLoading {MODEL_ID} (4-bit BnB quantization)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
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
            temp_rows = run_temp_sweep(llm, tokenizer, data, args.prompt, judger, out_dir, args.batch_size)
        else:
            n = int(exp.replace("voting_n", ""))
            voting_rows.append(run_voting(n, llm, tokenizer, data, args.prompt, judger, out_dir, args.batch_size))

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
