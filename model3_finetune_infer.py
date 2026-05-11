"""
Model 3 — Inference: QLoRA Fine-tuned Model for CSE 151B Math Competition

Loads Qwen/Qwen3-4B-Thinking-2507 with a fine-tuned LoRA adapter and runs
inference on the competition dataset (public or private).

The base model is loaded in 4-bit quantization (same as training) and the
LoRA adapter from the specified checkpoint is merged for inference.

Usage
-----
  python model3_finetune_infer.py [options]

  --checkpoint DIR   Path to LoRA adapter directory from model3_finetune_train.py
                     (default: checkpoints/model3_qlora)
  --data PATH        Dataset to run inference on (default: data/public.jsonl)
                     Use data/private.jsonl to generate Kaggle submission
  --limit N          Only run on first N questions
  --gpu ID           CUDA_VISIBLE_DEVICES (default: 0)
  --no_eval          Skip scoring (use for private test set where answers are unknown)
  --output DIR       Directory for output files (default: results/)

Output (public set, with answers)
----------------------------------
  results/model3_finetune_results.jsonl    — per-question records with correctness
  results/model3_submission.csv            — Kaggle-format submission (all questions)

Output (private set, --no_eval)
---------------------------------
  results/model3_private_submission.csv    — Kaggle-format submission

Runtime estimate (A100 40 GB)
------------------------------
  Full 1126-question public set: ~12 min
  Full private set:              ~12 min
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_ID           = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_CHECKPOINT = "checkpoints/model3_qlora"
DATA_PATH          = "data/public.jsonl"
MAX_TOKENS         = 1024

SAMPLING_PARAMS = dict(
    max_new_tokens=MAX_TOKENS,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    repetition_penalty=1.0,
    do_sample=True,
)

SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Show all your reasoning, then put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside "
    "a single \\boxed{}, e.g. \\boxed{3, 7}."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

# ── Model Loading ──────────────────────────────────────────────────────────────

def load_model_with_adapter(checkpoint_dir: str):
    """
    Load the 4-bit base model and apply the LoRA adapter from checkpoint_dir.
    Falls back to base model only if no adapter is found (with a warning).
    """
    try:
        from peft import PeftModel
        has_peft = True
    except ImportError:
        has_peft = False
        print("WARNING: peft not installed. Loading base model without adapter.")
        print("         Install with: pip install peft")

    print(f"Loading {MODEL_ID} (4-bit BnB quantization)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )

    adapter_path = Path(checkpoint_dir)
    adapter_config = adapter_path / "adapter_config.json"

    if has_peft and adapter_config.exists():
        print(f"Loading LoRA adapter from: {checkpoint_dir}")
        model = PeftModel.from_pretrained(model, checkpoint_dir)
        model.eval()
        print("Adapter loaded successfully.")
    elif has_peft:
        print(f"WARNING: No adapter_config.json found in {checkpoint_dir}.")
        print("         Running with base model only (no fine-tuning applied).")
    else:
        print("Running with base model only.")

    return model, tokenizer

# ── Prompt Building ────────────────────────────────────────────────────────────

def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT, question

# ── Scoring Helpers ────────────────────────────────────────────────────────────

def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_response(item: dict, response: str, judger) -> bool:
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

# ── Inference ─────────────────────────────────────────────────────────────────

def generate_response(model, tokenizer, item: dict) -> str:
    sys_p, usr_p = build_prompt(item["question"], item.get("options"))
    prompt_text  = tokenizer.apply_chat_template(
        [{"role": "system", "content": sys_p},
         {"role": "user",   "content": usr_p}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        prompt_text, return_tensors="pt", truncation=True, max_length=8192
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **SAMPLING_PARAMS)

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ── CSV Submission Writer ─────────────────────────────────────────────────────

def write_submission_csv(results: list[dict], csv_path: Path):
    """Write Kaggle submission CSV with id and response columns."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "response"])
        for r in results:
            writer.writerow([r["id"], r["response"]])
    print(f"Submission CSV: {csv_path}  ({len(results)} rows)")

# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Model 3 Inference: Fine-tuned QLoRA — CSE 151B Competition"
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="LoRA adapter checkpoint directory")
    parser.add_argument("--data",       default=DATA_PATH,
                        help="Input JSONL (public.jsonl or private.jsonl)")
    parser.add_argument("--limit",      type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS,
                        help=f"Max new tokens per response (default: {MAX_TOKENS}). "
                             "Use 2048 for fast smoke tests.")
    parser.add_argument("--gpu",        default="0")
    parser.add_argument("--no_eval",    action="store_true",
                        help="Skip scoring (use for private test set)")
    parser.add_argument("--output",     default="results",
                        help="Output directory (default: results/)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    SAMPLING_PARAMS["max_new_tokens"] = args.max_tokens

    data = [json.loads(line) for line in open(args.data)]
    if args.limit:
        data = data[: args.limit]

    is_private = args.no_eval or not any("answer" in d for d in data)
    n_mcq      = sum(bool(d.get("options")) for d in data)
    print(f"Dataset   : {args.data}")
    print(f"Questions : {len(data)}  ({n_mcq} MCQ, {len(data)-n_mcq} free-form)")
    print(f"Mode      : {'inference-only (no ground truth)' if is_private else 'eval'}")
    print(f"Checkpoint: {args.checkpoint}")

    model, tokenizer = load_model_with_adapter(args.checkpoint)

    if not is_private:
        sys.path.insert(0, str(Path(__file__).parent))
        from judger import Judger
        judger = Judger(strict_extract=False)
    else:
        judger = None

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    jsonl_name = "model3_finetune_results.jsonl" if not is_private else "model3_private_results.jsonl"
    jsonl_path = out_dir / jsonl_name

    # Resume: skip IDs already in the output file
    done_ids: set[int] = set()
    if jsonl_path.exists():
        for line in open(jsonl_path):
            try:
                done_ids.add(json.loads(line)["id"])
            except Exception:
                pass
    remaining = [d for d in data if d["id"] not in done_ids]
    if done_ids:
        print(f"Resuming: {len(done_ids)} done, {len(remaining)} remaining")

    # Write each result immediately so progress survives a killed session
    with open(jsonl_path, "a") as f_out:
        for item in tqdm(remaining, desc="Generating"):
            response = generate_response(model, tokenizer, item)
            record   = {
                "id":       item["id"],
                "is_mcq":   bool(item.get("options")),
                "response": response,
            }
            if not is_private and "answer" in item:
                record["gold"]    = item["answer"]
                record["correct"] = score_response(item, response, judger)
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()

    results = [json.loads(l) for l in open(jsonl_path)]
    print(f"Results JSONL: {jsonl_path}  ({len(results)} records)")

    # Save submission CSV
    csv_name = "model3_submission.csv" if not is_private else "model3_private_submission.csv"
    write_submission_csv(results, out_dir / csv_name)

    # Print accuracy if we have ground truth
    if not is_private and all("correct" in r for r in results):
        mcq_res  = [r for r in results if r["is_mcq"]]
        free_res = [r for r in results if not r["is_mcq"]]

        def acc(s):
            return sum(r["correct"] for r in s) / len(s) * 100 if s else 0.0

        W = 60
        print(f"\n{'='*W}")
        print(f"  RESULTS — Model 3: Fine-tuned QLoRA  ({len(results)} questions)")
        print(f"{'='*W}")
        print(f"  Checkpoint : {args.checkpoint}")
        print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d}/{len(mcq_res):4d}"
              f"  ({acc(mcq_res):.2f}%)")
        print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d}/{len(free_res):4d}"
              f"  ({acc(free_res):.2f}%)")
        print(f"  Overall    : {sum(r['correct'] for r in results):4d}/{len(results):4d}"
              f"  ({acc(results):.2f}%)")
        print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
