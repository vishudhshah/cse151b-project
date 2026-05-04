"""
Model 3 — Training: QLoRA Supervised Fine-Tuning for CSE 151B Math Competition

Fine-tunes Qwen/Qwen3-4B-Thinking-2507 on competition-level math problems using
QLoRA (4-bit base weights + LoRA adapters) and HuggingFace SFTTrainer.

Dataset
-------
  Primary  : lighteval/MATH  (7,500 competition math problems with detailed solutions)
  Fallback : hendrycks/competition_math

Training format
---------------
  The MATH dataset provides step-by-step solutions that end with \\boxed{answer}.
  Each training sample is formatted as a Qwen3 chat turn:
    system  : math expert system prompt
    user    : problem statement
    assistant: full reference solution (which ends with \\boxed{answer})
  Loss is computed on the FULL sequence (system + user + assistant) for simplicity.
  The model learns to produce detailed CoT solutions ending in \\boxed{}.

QLoRA setup
-----------
  Base model : 4-bit NF4 quantization (BitsAndBytes)
  LoRA rank  : 16, alpha: 32, dropout: 0.05
  Targets    : q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  Optimizer  : paged_adamw_8bit, lr=2e-4, cosine schedule, warmup_ratio=0.03
  Effective batch: 8  (batch_size=1 × grad_accumulation=8)
  Epochs     : 3  (configurable via --epochs)
  Max seq len: 4096

Usage
-----
  # Install extra dependencies first (not in starter venv):
  # pip install trl peft datasets accelerate bitsandbytes

  python model3_finetune_train.py [options]

  --epochs N         Number of training epochs (default: 3)
  --max_steps N      Override epochs with a fixed step count (for smoke tests)
  --lora_rank R      LoRA rank (default: 16)
  --lr FLOAT         Learning rate (default: 2e-4)
  --output DIR       Checkpoint output directory (default: checkpoints/model3_qlora)
  --gpu ID           CUDA_VISIBLE_DEVICES (default: 0)
  --hf_cache DIR     Override HF cache dir (set to scratch on DataHub to save disk)
  --subset N         Only use first N training examples (for quick debugging)

Output
------
  checkpoints/model3_qlora/          — LoRA adapter checkpoints
  checkpoints/model3_qlora/adapter_model.safetensors  — final adapter weights
  checkpoints/model3_qlora/training_log.jsonl         — per-step loss log

Runtime estimate (A100 40 GB)
------------------------------
  Full 3-epoch run (7,500 samples × 3): ~5–8 hours
  Quick smoke test (--max_steps 20)  : ~5 minutes
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# ── Dependency check ───────────────────────────────────────────────────────────
_MISSING = []
for pkg in ["trl", "peft", "datasets"]:
    try:
        __import__(pkg)
    except ImportError:
        _MISSING.append(pkg)
if _MISSING:
    print("ERROR: Missing dependencies. Install with:")
    print(f"  pip install {' '.join(_MISSING)}")
    sys.exit(1)

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_ID      = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_OUTPUT = "checkpoints/model3_qlora"

SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Show all your reasoning, then put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside "
    "a single \\boxed{}, e.g. \\boxed{3, 7}."
)

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Dataset Loading ────────────────────────────────────────────────────────────

def load_math_dataset(subset: int = None):
    """
    Load MATH training data. Tries two sources in order:

    1. EleutherAI/hendrycks_math — same 7,500 problems as the original MATH
       dataset but in standard Parquet format (no loading scripts required).
       Loads all 7 subjects and concatenates them.
    2. HuggingFaceH4/MATH-500 — 500-problem evaluation set; used as a fallback
       if the full dataset is unreachable.
    """
    from datasets import concatenate_datasets
    print("Loading MATH training dataset...")

    # ── Option 1: full MATH dataset (7,500 problems across 7 subjects) ──────────
    SUBJECTS = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    ]
    try:
        splits = [
            load_dataset("EleutherAI/hendrycks_math", name=s, split="train")
            for s in SUBJECTS
        ]
        ds = concatenate_datasets(splits)
        print(f"  Loaded {len(ds)} examples from EleutherAI/hendrycks_math (all subjects)")
        if subset:
            ds = ds.select(range(min(subset, len(ds))))
            print(f"  Using first {len(ds)} examples (--subset {subset})")
        return ds
    except Exception as e:
        print(f"  EleutherAI/hendrycks_math failed: {e}")

    # ── Option 2: MATH-500 fallback ──────────────────────────────────────────────
    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        print(f"  Loaded {len(ds)} examples from HuggingFaceH4/MATH-500 (fallback)")
        if subset:
            ds = ds.select(range(min(subset, len(ds))))
        return ds
    except Exception as e:
        print(f"  HuggingFaceH4/MATH-500 failed: {e}")

    raise RuntimeError(
        "Could not load any MATH dataset. Check HuggingFace Hub access "
        "and that HF_DATASETS_OFFLINE is not set to 1."
    )

# ── Data Formatting ────────────────────────────────────────────────────────────

def extract_boxed_answer(solution: str) -> str:
    """Pull the content of the last \\boxed{} from a solution string."""
    idx = solution.rfind("\\boxed{")
    if idx < 0:
        return ""
    depth = 0
    i = idx + len("\\boxed{")
    start = i
    while i < len(solution):
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            if depth == 0:
                return solution[start:i]
            depth -= 1
        i += 1
    return ""


def format_sample(sample: dict, tokenizer) -> str:
    """
    Format one MATH dataset sample as a full Qwen3 chat string.

    The assistant turn is the raw reference solution from the MATH dataset,
    which naturally includes step-by-step reasoning and ends with \\boxed{answer}.
    Training loss is computed over the full sequence.
    """
    # Different dataset versions use different field names
    problem  = sample.get("problem")  or sample.get("question",  "")
    solution = sample.get("solution") or sample.get("answer",    "")

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": problem},
        {"role": "assistant", "content": solution},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

# ── Loss-logging Callback ──────────────────────────────────────────────────────

class LossLogger(TrainerCallback):
    """Appends {step, loss} records to a JSONL file for post-hoc analysis."""

    def __init__(self, log_path: Path):
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"step": state.global_step, "loss": logs["loss"]}) + "\n")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Model 3 Training: QLoRA SFT — CSE 151B Competition"
    )
    parser.add_argument("--epochs",    type=int,   default=3)
    parser.add_argument("--max_steps", type=int,   default=-1,
                        help="Override epochs (use for smoke tests, e.g. --max_steps 20)")
    parser.add_argument("--lora_rank", type=int,   default=16)
    parser.add_argument("--lr",        type=float, default=2e-4)
    parser.add_argument("--output",    default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu",       default="0")
    parser.add_argument("--hf_cache",  default=None,
                        help="Override HuggingFace cache dir (e.g., /scratch/$USER/hf_cache)")
    parser.add_argument("--subset",    type=int,   default=None,
                        help="Use only first N training examples (for debugging)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache
        print(f"HF cache: {args.hf_cache}")

    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"

    # ── Load tokenizer ─────────────────────────────────────────────────────────
    print(f"Loading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token     = tokenizer.eos_token
    tokenizer.padding_side  = "right"   # required for SFTTrainer's causal-mask logic

    # ── Load dataset ───────────────────────────────────────────────────────────
    raw_dataset = load_math_dataset(subset=args.subset)

    # Format all samples into full chat strings
    def tokenize_fn(batch):
        texts = [format_sample(s, tokenizer) for s in
                 [{k: batch[k][i] for k in batch} for i in range(len(batch[list(batch.keys())[0]]))]]
        return {"text": texts}

    dataset = raw_dataset.map(
        tokenize_fn, batched=True, remove_columns=raw_dataset.column_names,
        desc="Formatting dataset",
    )
    print(f"Dataset ready: {len(dataset)} training samples")
    print(f"Sample (first 300 chars):\n{dataset[0]['text'][:300]}\n")

    # ── Load base model (4-bit) ────────────────────────────────────────────────
    print(f"Loading {MODEL_ID} with 4-bit quantization...")
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
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    print("Base model loaded.")

    # ── LoRA config ────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,   # alpha = 2 × rank is a common heuristic
        lora_dropout=0.05,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training arguments (SFTConfig = TrainingArguments + SFT-specific fields) ─
    training_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,          # -1 means "use epochs"
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,     # effective batch = 8
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        optim="paged_adamw_8bit",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",                  # disable wandb/mlflow for DataHub
        dataloader_num_workers=0,
        remove_unused_columns=False,
        dataset_text_field="text",         # moved here from SFTTrainer in trl 1.x
        max_length=4096,                   # renamed from max_seq_length in trl 1.1+
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,        # replaces 'tokenizer' in trl 1.x
        train_dataset=dataset,
        args=training_args,
        callbacks=[LossLogger(log_path)],
    )

    print("\n" + "="*60)
    print("Training configuration")
    print("="*60)
    print(f"  Model         : {MODEL_ID}")
    print(f"  LoRA rank     : {args.lora_rank}  alpha: {args.lora_rank*2}")
    print(f"  LR            : {args.lr}  scheduler: cosine  warmup: 3%")
    print(f"  Effective bs  : {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Epochs        : {args.epochs}  max_steps: {args.max_steps}")
    print(f"  Max seq len   : 4096")
    print(f"  Training set  : {len(dataset)} samples")
    print(f"  Output        : {out_dir}")
    print(f"  Loss log      : {log_path}")
    print("="*60 + "\n")

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    print(f"\nFine-tuning complete. Adapter saved to: {out_dir}")
    print(f"Training loss log  : {log_path}")
    print(f"\nNext step: run inference with\n"
          f"  python model3_finetune_infer.py --checkpoint {out_dir}")


if __name__ == "__main__":
    main()
