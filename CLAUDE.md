# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kaggle competition: improve mathematical reasoning of **Qwen/Qwen3-4B-Thinking-2507** (4B-param thinking model) using model-intrinsic techniques only. Three approaches explored: prompt engineering, sampling/majority voting, and QLoRA fine-tuning. Scored on unified accuracy (correct/total) across 1,126 math problems in `data/public.jsonl`.

## Commands

### Environment (DataHub)
```bash
# New pod — create venv inheriting pre-installed system packages
uv venv .venv --seed --system-site-packages
source .venv/bin/activate
pip install -r requirements.txt

# Returning to same pod — just activate
source .venv/bin/activate

export HF_TOKEN=hf_your_token_here    # avoids HuggingFace rate limits
export HF_HOME=/datasets/$USER/hf_cache  # avoids home dir disk quota during training
```

### Running experiments
```bash
# Smoke tests (~2 min total)
python model1_prompt_engineering.py --variant v0_baseline --limit 5 --max_tokens 2048
python model2_sampling_voting.py --experiment voting_n3 --limit 5 --max_tokens 2048
python model3_finetune_train.py --max_steps 5 --subset 50
python model3_finetune_infer.py --checkpoint checkpoints/model3_qlora --limit 5 --max_tokens 4096

# Model 1 — all 4 prompt variants (~40 min on A30)
python model1_prompt_engineering.py --variant all
python model1_prompt_engineering.py --variant v2_fewshot  # single variant (~10 min)

# Model 2 — temperature sweep or majority voting
python model2_sampling_voting.py --experiment temp_sweep
python model2_sampling_voting.py --experiment voting_n5

# Model 3 — training then inference
python model3_finetune_train.py --epochs 3
python model3_finetune_infer.py --checkpoint checkpoints/model3_qlora
```

### Evaluate results
```python
import json

def load_acc(path):
    results = [json.loads(l) for l in open(path)]
    m = [r for r in results if r["is_mcq"]]; f = [r for r in results if not r["is_mcq"]]
    p = lambda s: sum(r["correct"] for r in s)/len(s)*100 if s else 0
    return p(m), p(f), p(results), len(results)
```

### Kaggle submission (private test set)
```bash
python model3_finetune_infer.py --checkpoint checkpoints/model3_qlora --data data/private.jsonl --no_eval
```

## Architecture

### Core files (do not modify)
- **`judger.py`** — `Judger` class handles answer equivalence via SymPy (symbolic math), LaTeX normalization, degree normalization, and inverse trig normalization. `Judger.auto_judge(pred, gold, options)` is the primary scoring function.
- **`utils.py`** — constants and helper functions used by `judger.py`: LaTeX stripping, boxed extraction (`last_boxed_only_string`, `remove_boxed`), fraction/sqrt normalization.

### Model scripts
The model outputs `<think>...</think>` reasoning before the final answer; the judger strips everything before `</think>` and looks for `\boxed{}`.

- **`model1_prompt_engineering.py`** — 4 prompt variants (v0 baseline, v1 enhanced CoT, v2 few-shot, v3 self-verification). Uses **vLLM** (bfloat16, no quantization) for inference. Sampling fixed across all variants (`T=0.6, top_p=0.95, top_k=20`, `thinking_budget=3072`, `max_tokens=4096`). Flushes results every 50 questions for resume support. Writes `results/model1_<variant>_results.jsonl`.

- **`model2_sampling_voting.py`** — Temperature sweep (T=0.0–0.9) and majority voting (N=3,5,7 at T=0.7). Uses **vLLM** with `SamplingParams(n=N)` so all N voting samples are generated in a single engine call per chunk (much faster than N sequential calls). Voting extracts `\boxed{}` from each sample, normalizes via `judger.norm_ans_str()`, and takes the modal answer. Writes `results/model2_<experiment>_results.jsonl`.

- **`model3_finetune_train.py`** — QLoRA fine-tuning on `lighteval/MATH` (7,500 problems). Frozen 4-bit base (BitsAndBytes NF4) + trainable LoRA adapters (rank=16, alpha=32) on attention + FFN layers. Uses `paged_adamw_8bit`, lr=2e-4, cosine schedule, effective batch=8, 3 epochs. Training samples are formatted with the MATH solution inside `<think>...</think>` and the `\boxed{}` answer outside, matching the model's inference-time output format. Loss is computed on the full sequence (TRL 1.4.0 removed `DataCollatorForCompletionOnlyLM`). Max sequence length: 16384. Saves to `checkpoints/model3_qlora/`.

- **`model3_finetune_infer.py`** — Loads base model with **vLLM** (bfloat16) + LoRA adapter via `LoRARequest`. Falls back to base model if no `adapter_config.json` is found. Uses `enable_thinking=True` in `apply_chat_template`. `max_tokens=4096` default (`thinking_budget=3072` + answer). Flushes every 50 questions. Writes `.jsonl` results and `results/model3_submission.csv` for Kaggle.

### Result file format
All `.jsonl` files in `results/` have one JSON object per line with fields: `id`, `is_mcq`, `gold`, `response` (or `responses` for voting), `correct`. Voting records also have `voted`, `agreement`, `n_samples`.

### Resume behavior
All inference scripts write results to disk and skip already-completed IDs on re-run (model3 flushes after each batch). Training resumes from the last epoch checkpoint automatically.

## Data

- `data/public.jsonl` — 1,126 labeled questions (IDs 0–1125). Fields: `id`, `question` (LaTeX), `answer` (string or list), `options` (list, MCQ only).
- `data/private.jsonl` — test set without answers; for Kaggle submission.
- MCQ: answer is a single letter; model must output `\boxed{C}`.
- Free-form multi-answer: multiple `[ANS]` placeholders; model outputs `\boxed{3, 7}`; ALL sub-answers must be correct.

## DataHub Workflow

```bash
# 1. SSH in
ssh dsmlp

# 2. Set pod timeout (required for Model 3 training)
export K8S_TIMEOUT_SECONDS=43200

# 3. Launch GPU pod
launch-sp26-cuda128.sh -l gpu-class=medium -W CSE151B_SP26_A00 -g 1 -c 16 -m 64

# 4. In the pod
cd 'private/CSE 151B/cse151b-project'
source .venv/bin/activate
export HF_TOKEN=<your token>   # from huggingface.co/settings/tokens
# then run model commands
```

Model 3 training (~10 hours) must run alone — parallel inference jobs cause GPU contention.

Git on DataHub uses HTTPS (no SSH keys on pods):
```bash
git config --global credential.helper store
git remote set-url origin https://github.com/vishudhshah/cse151b-project.git
```
