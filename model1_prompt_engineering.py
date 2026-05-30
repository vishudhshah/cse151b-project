"""
Model 1: Prompt Engineering for CSE 151B Math Reasoning Competition

Compares 4 system-prompt strategies on Qwen/Qwen3-4B-Thinking-2507:
  v0_baseline     — exact starter-code prompts (control)
  v1_enhanced_cot — explicit numbered step-by-step CoT instructions
  v2_fewshot      — 2 MCQ + 2 free-form worked examples prepended to user turn
  v3_verification — solve first, then verify before committing to final answer

All 4 variants use the same model weights and sampling parameters so that any
accuracy difference is attributable to the prompt alone.

Usage
-----
  python model1_prompt_engineering.py [options]

  --variant   {v0_baseline,v1_enhanced_cot,v2_fewshot,v3_verification,all}
              Default: all
  --limit N   Only run on first N questions (useful for smoke tests)
  --gpu ID    CUDA_VISIBLE_DEVICES (default: 0)
  --data PATH Path to JSONL dataset (default: data/public.jsonl)

Output
------
  results/model1_{variant}_results.jsonl  — per-question records
  Printed summary table comparing all variants

Runtime estimate (A30 24 GB, vLLM, bfloat16)
---------------------------------------------
  ~10 min / variant on full 1126-question public set
  ~1 min  / variant on --limit 5 (smoke test)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_ID   = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH  = "data/public.jsonl"
MAX_TOKENS = 4096
# Cap the thinking block. Remaining tokens go to the answer.
THINKING_BUDGET = 3072

# Sampling kept identical across all variants to isolate prompt effect.
_SAMPLING_KWARGS = dict(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    repetition_penalty=1.0,
)

# ── Prompt Definitions ─────────────────────────────────────────────────────────

# v0 — exact starter-code prompts
_V0_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)
_V0_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

# v1 — numbered steps make CoT structure explicit, reducing skipped reasoning
_V1_MATH = (
    "You are an expert mathematician. Solve the problem using the following structure:\n"
    "1. Identify what is being asked and what information is given.\n"
    "2. Select the relevant formulas or theorems.\n"
    "3. Carry out each calculation step, showing your work.\n"
    "4. Verify your answer (e.g., substitute back or sanity-check units/order-of-magnitude).\n"
    "5. Write your final answer inside \\boxed{}.\n"
    "If the problem has multiple [ANS] placeholders, separate all answers by commas "
    "inside a single \\boxed{}, e.g. \\boxed{3, 7}."
)
_V1_MCQ = (
    "You are an expert mathematician.\n"
    "1. Read the question and understand what it asks.\n"
    "2. Compute or reason about the answer independently before looking at the choices.\n"
    "3. Match your result to the best option; rule out incorrect choices with brief reasoning.\n"
    "4. Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

# v2 — few-shot: same system prompt as v1; examples prepended to the user turn
_V2_MATH = _V1_MATH
_V2_MCQ  = _V1_MCQ

# Two free-form worked examples that demonstrate the expected chain-of-thought and \boxed{} format
_FEWSHOT_FREEFORM = """\
Here are two solved example problems showing the expected format.

--- Example 1 ---
Problem: Solve $3x - 7 = 14$. $x =$ [ANS]
Step 1: Add 7 to both sides: $3x = 21$
Step 2: Divide by 3: $x = 7$
Answer: \\boxed{7}

--- Example 2 ---
Problem: Find all real solutions to $x^2 - 5x + 6 = 0$. Solutions: [ANS]
Step 1: Factor the quadratic: $(x - 2)(x - 3) = 0$
Step 2: Set each factor to zero: $x = 2$ or $x = 3$
Answer: \\boxed{2, 3}

--- Now solve the following problem ---
"""

# Two MCQ worked examples
_FEWSHOT_MCQ = """\
Here are two solved example problems showing the expected format.

--- Example 1 ---
Problem: What is the value of $\\binom{5}{2}$?
Options:
A. 5
B. 10
C. 15
D. 20
Solution: $\\binom{5}{2} = \\frac{5!}{2!\\,3!} = \\frac{20}{2} = 10$
Correct option: B
Answer: \\boxed{B}

--- Example 2 ---
Problem: The derivative of $f(x) = \\sin(x)$ is:
Options:
A. $-\\cos(x)$
B. $\\tan(x)$
C. $\\cos(x)$
D. $-\\sin(x)$
Solution: By standard differentiation, $\\frac{d}{dx}[\\sin x] = \\cos x$
Correct option: C
Answer: \\boxed{C}

--- Now solve the following problem ---
"""

# v3 — verification: model is instructed to re-check before writing the final answer
_V3_MATH = (
    "You are an expert mathematician.\n"
    "Step 1 — Solve: Work through the problem step by step.\n"
    "Step 2 — Verify: Check your answer by an independent method "
    "(e.g., substitute back into the equation, check limiting cases, or redo a key calculation).\n"
    "Step 3 — Correct if needed and write your final verified answer inside \\boxed{}.\n"
    "For multiple sub-answers, separate them by commas inside a single \\boxed{}."
)
_V3_MCQ = (
    "You are an expert mathematician.\n"
    "Step 1 — Solve: Reason through the problem independently.\n"
    "Step 2 — Verify: Confirm that your answer is consistent with the given information "
    "and that the other options can be ruled out.\n"
    "Step 3 — Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

# Registry of all variants
VARIANTS: dict[str, dict] = {
    "v0_baseline": {
        "sys_math": _V0_MATH,
        "sys_mcq":  _V0_MCQ,
        "prefix_math": "",
        "prefix_mcq":  "",
        "description": "Exact starter-code prompts (control)",
    },
    "v1_enhanced_cot": {
        "sys_math": _V1_MATH,
        "sys_mcq":  _V1_MCQ,
        "prefix_math": "",
        "prefix_mcq":  "",
        "description": "Explicit numbered CoT steps",
    },
    "v2_fewshot": {
        "sys_math": _V2_MATH,
        "sys_mcq":  _V2_MCQ,
        "prefix_math": _FEWSHOT_FREEFORM,
        "prefix_mcq":  _FEWSHOT_MCQ,
        "description": "2 worked examples in user turn",
    },
    "v3_verification": {
        "sys_math": _V3_MATH,
        "sys_mcq":  _V3_MCQ,
        "prefix_math": "",
        "prefix_mcq":  "",
        "description": "Solve then verify before final answer",
    },
}

# ── Prompt Building ────────────────────────────────────────────────────────────

def build_prompt(question: str, options: Optional[list], cfg: dict) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for one question and one variant config."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        user = cfg["prefix_mcq"] + f"{question}\n\nOptions:\n{opts_text}"
        return cfg["sys_mcq"], user
    return cfg["sys_math"], cfg["prefix_math"] + question

# ── Scoring Helpers ────────────────────────────────────────────────────────────

def extract_letter(text: str) -> str:
    """Extract the letter from \\boxed{X}; fall back to last capital letter."""
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_response(item: dict, response: str, judger) -> bool:
    """Return True if the model response is correct for this item."""
    is_mcq = bool(item.get("options"))
    gold   = item["answer"]
    if is_mcq:
        return extract_letter(response) == str(gold).strip().upper()
    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        return judger.auto_judge(
            pred=response,
            gold=gold_list,
            options=[[]] * len(gold_list),
        )
    except Exception:
        return False

# ── Inference ─────────────────────────────────────────────────────────────────

def _format_prompts(tokenizer, items: list[dict], cfg: dict) -> list[str]:
    prompt_texts = []
    for item in items:
        sys_p, usr_p = build_prompt(item["question"], item.get("options"), cfg)
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


def generate_chunk(llm, tokenizer, items: list[dict], cfg: dict,
                   sampling_params: SamplingParams) -> list[str]:
    prompt_texts = _format_prompts(tokenizer, items, cfg)
    outputs = llm.generate(prompt_texts, sampling_params)
    return [o.outputs[0].text.strip() for o in outputs]

# ── Per-Variant Runner ─────────────────────────────────────────────────────────

_CHUNK_SIZE = 50  # flush results to disk every N questions

def run_variant(variant: str, llm, tokenizer, sampling_params: SamplingParams,
                data: list, judger, out_dir: Path) -> dict:
    cfg      = VARIANTS[variant]
    out_path = out_dir / f"model1_{variant}_results.jsonl"

    # Resume: load IDs already written to disk
    done_ids: set[int] = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done_ids.add(json.loads(line)["id"])
            except Exception:
                pass

    remaining = [d for d in data if d["id"] not in done_ids]

    print(f"\n{'─'*60}")
    print(f"  Variant : {variant}  ({cfg['description']})")
    if done_ids:
        print(f"  Resuming: {len(done_ids)} already done, {len(remaining)} remaining")
    print(f"{'─'*60}")

    with open(out_path, "a") as f_out:
        with tqdm(total=len(remaining), desc=f"  {variant}") as pbar:
            for chunk_start in range(0, len(remaining), _CHUNK_SIZE):
                chunk = remaining[chunk_start : chunk_start + _CHUNK_SIZE]
                responses = generate_chunk(llm, tokenizer, chunk, cfg, sampling_params)
                for item, response in zip(chunk, responses):
                    record = {
                        "id":       item["id"],
                        "variant":  variant,
                        "is_mcq":   bool(item.get("options")),
                        "gold":     item["answer"],
                        "response": response,
                        "correct":  score_response(item, response, judger),
                    }
                    f_out.write(json.dumps(record) + "\n")
                pbar.update(len(chunk))
                f_out.flush()

    # Read all results (existing + just written) for reporting
    results = [json.loads(l) for l in open(out_path)]

    mcq_res  = [r for r in results if r["is_mcq"]]
    free_res = [r for r in results if not r["is_mcq"]]

    def acc(subset: list) -> float:
        return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

    row = {
        "variant":   variant,
        "n_mcq":     len(mcq_res),
        "n_free":    len(free_res),
        "n_total":   len(results),
        "mcq_acc":   acc(mcq_res),
        "free_acc":  acc(free_res),
        "total_acc": acc(results),
    }
    print(f"  MCQ       : {sum(r['correct'] for r in mcq_res):4d}/{row['n_mcq']:4d} ({row['mcq_acc']:.2f}%)")
    print(f"  Free-form : {sum(r['correct'] for r in free_res):4d}/{row['n_free']:4d} ({row['free_acc']:.2f}%)")
    print(f"  Overall   : {sum(r['correct'] for r in results):4d}/{row['n_total']:4d} ({row['total_acc']:.2f}%)")
    print(f"  Saved to  : {out_path}")
    return row

# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Model 1: Prompt Engineering — CSE 151B Competition"
    )
    parser.add_argument("--variant", default="all",
                        choices=list(VARIANTS) + ["all"],
                        help="Prompt variant to run (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate on first N questions")
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS,
                        help=f"Max new tokens per response (default: {MAX_TOKENS}). "
                             "Use 2048 for smoke tests.")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES (default: 0)")
    parser.add_argument("--data", default=DATA_PATH, help="Path to JSONL dataset")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load dataset
    data = [json.loads(line) for line in open(args.data)]
    if args.limit:
        data = data[: args.limit]
    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = len(data) - n_mcq
    print(f"Dataset  : {args.data}")
    print(f"Questions: {len(data)}  ({n_mcq} MCQ, {n_free} free-form)")

    # Load tokenizer (for chat template formatting only)
    print(f"\nLoading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Load model with vLLM — bfloat16, no quantization needed for 4B on 24 GB+
    print(f"Loading {MODEL_ID} with vLLM (bfloat16)...")
    llm = LLM(
        model=MODEL_ID,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    print("Model loaded.\n")

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        **_SAMPLING_KWARGS,
    )

    # Load judger
    sys.path.insert(0, str(Path(__file__).parent))
    from judger import Judger
    judger = Judger(strict_extract=False)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    variants_to_run = list(VARIANTS) if args.variant == "all" else [args.variant]
    summary = [
        run_variant(v, llm, tokenizer, sampling_params, data, judger, out_dir)
        for v in variants_to_run
    ]

    # Summary table
    W = 72
    print(f"\n{'='*W}")
    print(f"  SUMMARY — Model 1: Prompt Engineering  ({len(data)} questions)")
    print(f"{'='*W}")
    print(f"  {'Variant':<22} {'MCQ':>8} {'Free-form':>12} {'Overall':>10}")
    print(f"  {'-'*(W-4)}")
    for row in summary:
        print(f"  {row['variant']:<22} {row['mcq_acc']:>7.2f}%"
              f" {row['free_acc']:>11.2f}% {row['total_acc']:>9.2f}%")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
