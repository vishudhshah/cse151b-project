# CSE 151B Spring 2026 — Project Guide

**Competition**: Pushing the Boundary of LLM Mathematical Reasoning  
**Team**: Rohan Raval · Tanishq Rathore · Vishudh Shah  
**Model**: `Qwen/Qwen3-4B-Thinking-2507` (4B parameters, thinking/reasoning model)

This guide is written for teammates filling in the milestone report. It explains what every script does, how the models work, what the results mean, and which numbers to put where in the LaTeX.

---

## Table of Contents

1. [Competition Overview](#1-competition-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dataset](#3-dataset)
4. [Evaluation Metric](#4-evaluation-metric)
5. [Baseline — Starter Code](#5-baseline--starter-code)
6. [Model 1 — Prompt Engineering](#6-model-1--prompt-engineering)
7. [Model 2 — Sampling & Majority Voting](#7-model-2--sampling--majority-voting)
8. [Model 3 — QLoRA Fine-Tuning](#8-model-3--qlora-fine-tuning)
9. [Reading Result Files](#9-reading-result-files)
10. [Running on DataHub](#10-running-on-datahub)
    - [Pre-flight checks](#pre-flight-checks)
    - [Resume behaviour](#resume-behaviour)
    - [Time budget](#time-budget)
    - [Step-by-step commands](#step-by-step-commands)
    - [Getting all numbers at once](#getting-all-numbers-at-once)
    - [Background job management](#background-job-management)
    - [Troubleshooting](#troubleshooting)
11. [Milestone Report Writing Guide](#11-milestone-report-writing-guide)
12. [Generating a Kaggle Submission](#12-generating-a-kaggle-submission)

---

## 1. Competition Overview

The Kaggle competition asks us to improve the mathematical reasoning of **Qwen3-4B-Thinking-2507**, a 4-billion-parameter open-weight reasoning model, using only model-intrinsic techniques — no external APIs, no calculators at inference time.

Problems span high-school to graduate-level math across algebra, calculus, statistics, combinatorics, and more. The test set is a mix of:
- **Multiple-choice questions (MCQ)**: Model selects one letter (A–J) and outputs `\boxed{C}`.
- **Free-form questions**: Model computes a numerical or symbolic answer and outputs `\boxed{42}` or `\boxed{3, 7}` for multi-part problems.

**Scoring**: unified accuracy = (# correct) / (# total questions). All questions weighted equally regardless of difficulty or source.

---

## 2. Repository Structure

```
Project/
│
├── data/
│   ├── public.jsonl          # 1,126 questions WITH answers (for local evaluation)
│   └── len(public).txt       # Confirms: IDs 0–1125
│
├── results/                  # All model outputs land here (created at runtime)
│   ├── model1_v0_baseline_results.jsonl
│   ├── model1_v1_enhanced_cot_results.jsonl
│   ├── model1_v2_fewshot_results.jsonl
│   ├── model1_v3_verification_results.jsonl
│   ├── model2_temp0p0_results.jsonl      # Greedy decode
│   ├── model2_temp0p3_results.jsonl
│   ├── model2_temp0p5_results.jsonl
│   ├── model2_temp0p7_results.jsonl
│   ├── model2_temp0p9_results.jsonl
│   ├── model2_voting_n3_results.jsonl
│   ├── model2_voting_n5_results.jsonl
│   ├── model2_voting_n7_results.jsonl
│   ├── model3_finetune_results.jsonl
│   └── model3_submission.csv            # Kaggle submission CSV
│
├── checkpoints/
│   └── model3_qlora/
│       ├── adapter_config.json          # LoRA adapter metadata
│       ├── adapter_model.safetensors    # Fine-tuned weights
│       ├── training_log.jsonl           # Loss per step (for loss curve plot)
│       └── tokenizer files
│
├── Milestone Report/
│   └── latex/
│       ├── main.tex                     # ← EDIT THIS for the report
│       └── references.bib              # ← Add citations here
│
├── starter_code_cse151b_comp.ipynb      # Baseline (Jupyter notebook)
├── judger.py                            # Scoring engine — do not modify
├── utils.py                             # Math normalization helpers
│
├── model1_prompt_engineering.py         # Prompt engineering experiments
├── model2_sampling_voting.py            # Sampling parameter experiments
├── model3_finetune_train.py             # Fine-tuning training script
├── model3_finetune_infer.py             # Fine-tuning inference + submission CSV
│
├── PROJECT_GUIDE.md                     # ← You are here
└── README.md                            # Short index
```

---

## 3. Dataset

**File**: `data/public.jsonl` — 1,126 questions, one JSON object per line.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Unique identifier (0–1125) |
| `question` | string | LaTeX-formatted problem. `[ANS]` marks where answers go. |
| `answer` | string or list | Ground truth. Single letter for MCQ; list of strings for free-form. |
| `options` | list or absent | Present only for MCQ. List of LaTeX option strings. |

### Question types

**MCQ** — `options` field is present. The model must output `\boxed{C}` (a single capital letter).

```json
{
  "id": 4,
  "question": "Given u(x,y) = x^3 + 6x^2y ..., find f(z).",
  "options": ["$(6+4i)z^5$", "$(1-2i)z^3$", ...],
  "answer": "C"
}
```

**Free-form (single answer)** — `options` absent, one `[ANS]`.

```json
{
  "id": 3,
  "question": "Reduce the fraction $\\frac{25}{40}$. [ANS]",
  "answer": ["5/8"]
}
```

**Free-form (multiple answers)** — multiple `[ANS]` placeholders; `answer` is a list. ALL must be correct for the question to count.

```json
{
  "id": 2,
  "question": "A turkey is taken from an oven... (a) temperature after 45 min? [ANS] (b) when does it cool to 100°F? [ANS]",
  "answer": ["143.224...", "2.326..."]
}
```

---

## 4. Evaluation Metric

**Judger** (`judger.py`): The `Judger` class handles answer equivalence. It is not simple string matching — it handles:
- Symbolic equivalence via SymPy (e.g., `1/2` == `0.5` == `\frac{1}{2}`)
- LaTeX normalization (strips `\dfrac`, `\text{}`, spaces, etc.)
- Degree normalization (`30°` == `30`)
- Inverse trig (`sin^{-1}` == `arcsin`)
- Multi-answer matching (all sub-answers must be correct)

**MCQ scoring**: Extract the letter from `\boxed{X}` in the model's response. Compare to the gold letter (case-insensitive).

**Free-form scoring**: `Judger.auto_judge(pred=response, gold=["answer1", "answer2"], options=[[],[]])` returns `True/False`.

**Unified accuracy**: `sum(correct) / len(all_questions)` across MCQ and free-form combined.

---

## 5. Baseline — Starter Code

**File**: `starter_code_cse151b_comp.ipynb`

### What it does

The baseline notebook runs the unmodified Qwen3-4B-Thinking-2507 model with minimal prompts and default sampling parameters.

### Model loading

```python
# 4-bit quantization to fit on DataHub GPUs
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
```

INT4 quantization reduces VRAM from ~8 GB (BF16) to ~2.5 GB, making the model runnable on a single consumer GPU.

### System prompts (baseline)

| Question type | System prompt |
|--------------|---------------|
| Free-form | `"You are an expert mathematician. Solve the problem step-by-step. Put your final answer inside \boxed{}. If the problem has multiple sub-answers, separate them by commas inside a single \boxed{}, e.g. \boxed{3, 7}."` |
| MCQ | `"You are an expert mathematician. Read the problem and the answer choices below, then select the single best answer. Output ONLY the letter of your chosen option inside \boxed{}, e.g. \boxed{C}."` |

### Sampling parameters (baseline)

| Parameter | Value |
|-----------|-------|
| `temperature` | 0.6 |
| `top_p` | 0.95 |
| `top_k` | 20 |
| `repetition_penalty` | 1.0 |
| `max_new_tokens` | 32,768 |

### Answer format

The model is a "thinking" model — it outputs a long chain of reasoning inside `<think>...</think>` tags before giving the final answer. The judger strips everything before `</think>` and looks for `\boxed{}` in the remaining text.

---

## 6. Model 1 — Prompt Engineering

**File**: `model1_prompt_engineering.py`

### Goal

Test whether better-crafted system prompts improve accuracy without changing anything else (same model, same sampling parameters, same temperature). This isolates the prompt as the independent variable.

### Design

All 4 variants share identical model weights and sampling parameters (`T=0.6, top_p=0.95, top_k=20`). Any accuracy difference is attributable to the prompt alone.

### Variants

#### v0 — Baseline (control)
Exact starter-code prompts. Used as the reference point.

#### v1 — Enhanced Chain-of-Thought
Replaces the generic "solve step-by-step" instruction with an explicit numbered structure:
```
1. Identify what is being asked and what information is given.
2. Select the relevant formulas or theorems.
3. Carry out each calculation step, showing your work.
4. Verify your answer.
5. Write your final answer inside \boxed{}.
```
**Hypothesis**: Explicit step labels reduce the chance the model skips the verification step or writes an unsupported answer.

#### v2 — Few-shot Examples
Prepends two worked examples to the user turn (not as separate chat messages — as text within the same user message). The examples are:

*Free-form examples*:
- `3x - 7 = 14 → x = 7` (linear equation)
- `x² - 5x + 6 = 0 → x = 2, 3` (quadratic with multi-answer format)

*MCQ examples*:
- `C(5,2) = 10 → \boxed{B}` (combinatorics)
- `d/dx[sin x] = cos x → \boxed{C}` (calculus)

**Hypothesis**: Seeing the expected format (reasoning → boxed answer) in context improves format compliance and reasoning quality.

#### v3 — Self-Verification
System prompt explicitly instructs the model to solve, then verify (by substituting back, checking edge cases, or re-deriving), then correct if needed before writing the final answer.
**Hypothesis**: Verification catches arithmetic errors that CoT alone does not.

### Output files

```
results/model1_v0_baseline_results.jsonl
results/model1_v1_enhanced_cot_results.jsonl
results/model1_v2_fewshot_results.jsonl
results/model1_v3_verification_results.jsonl
```

Each line is a JSON record:
```json
{
  "id": 42,
  "variant": "v1_enhanced_cot",
  "is_mcq": false,
  "gold": ["7", "3"],
  "response": "<think>...\n</think>\n\n\\boxed{7, 3}",
  "correct": true
}
```

### Running

```bash
# Smoke test (first 20 questions, one variant)
python model1_prompt_engineering.py --variant v0_baseline --limit 20

# Full run, all 4 variants on complete public set
python model1_prompt_engineering.py --variant all

# Single variant
python model1_prompt_engineering.py --variant v2_fewshot
```

### Summary table (printed at end of run)

```
========================================================================
  SUMMARY — Model 1: Prompt Engineering  (1126 questions)
========================================================================
  Variant                MCQ     Free-form    Overall
  ──────────────────────────────────────────────────────────────────────
  v0_baseline          XX.XX%      XX.XX%     XX.XX%
  v1_enhanced_cot      XX.XX%      XX.XX%     XX.XX%
  v2_fewshot           XX.XX%      XX.XX%     XX.XX%
  v3_verification      XX.XX%      XX.XX%     XX.XX%
========================================================================
```

---

## 7. Model 2 — Sampling & Majority Voting

**File**: `model2_sampling_voting.py`

### Goal

Explore two complementary sampling strategies:
1. **Temperature sweep**: Find the optimal temperature for single-sample inference.
2. **Majority voting (self-consistency)**: Generate N independent samples and pick the most common answer, reducing variance.

### Background: Why Majority Voting Works

Stochastic decoding (temperature > 0) introduces randomness. When the model solves a problem correctly with probability `p`, generating N independent samples and picking the majority answer boosts accuracy to approximately:
```
P(majority correct) = Σ_{k=⌈N/2⌉}^{N} C(N,k) * p^k * (1-p)^(N-k)
```
For `p = 0.6, N = 5`: single sample = 60%, majority vote ≈ 68%. The gain is larger for problems where the model is "on the edge" (40–70% individual correctness).

### Experiments

#### Experiment A: Temperature Sweep

Five independent runs, each generating **one sample per question** at a different temperature:

| Run label | Temperature | Decoding |
|-----------|-------------|----------|
| `temp0p0` | 0.0 | Greedy (deterministic) |
| `temp0p3` | 0.3 | Low randomness |
| `temp0p5` | 0.5 | Moderate |
| `temp0p7` | 0.7 | Default voting temperature |
| `temp0p9` | 0.9 | High randomness |

Greedy decoding (`T=0`) uses `do_sample=False` and always picks the highest-probability token — it is deterministic and produces the same answer every run. Higher temperatures sample from a wider distribution, sometimes finding correct paths the greedy path misses, but also introducing more errors.

#### Experiment B: Majority Voting

Three runs at `T=0.7`, generating **N samples per question** and voting:

| Experiment | N samples | Vote rule |
|-----------|-----------|-----------|
| `voting_n3` | 3 | Majority (≥2/3) |
| `voting_n5` | 5 | Majority (≥3/5) |
| `voting_n7` | 7 | Majority (≥4/7) |

**MCQ voting**: Extract letter from each sample's `\boxed{}`, take the modal letter.

**Free-form voting**: Extract `\boxed{}` content from each sample → normalize with `judger.norm_ans_str()` → take the modal normalized string → wrap in `\boxed{}` for scoring. If no strict majority (>50%), fall back to the first sample's answer.

**Agreement rate** (also logged): fraction of the N samples that agree with the voted answer. High agreement = the model is confident; low agreement = uncertain.

### Output files

```
results/model2_temp0p0_results.jsonl     # one record per question
results/model2_temp0p7_results.jsonl
results/model2_voting_n5_results.jsonl
...
```

Each record:
```json
{
  "id": 7,
  "experiment": "voting_n5",
  "n_samples": 5,
  "temperature": 0.7,
  "is_mcq": true,
  "gold": "B",
  "responses": ["...", "...", "...", "...", "..."],
  "voted": "B",
  "agreement": 0.8,
  "correct": true
}
```

### Running

```bash
# Temperature sweep (5 temperatures × full dataset)
python model2_sampling_voting.py --experiment temp_sweep

# Voting with 5 samples
python model2_sampling_voting.py --experiment voting_n5

# All experiments
python model2_sampling_voting.py --experiment all

# Limit to first 50 questions for testing
python model2_sampling_voting.py --experiment voting_n3 --limit 50
```

---

## 8. Model 3 — QLoRA Fine-Tuning

**Files**: `model3_finetune_train.py` (training) · `model3_finetune_infer.py` (inference)

### Goal

Supervised fine-tuning (SFT) teaches the model to produce better math solutions by training it on a large set of high-quality worked examples. Unlike prompt engineering (which only changes the input) or sampling (which only changes the decoding), fine-tuning updates the model weights.

### Method: QLoRA

**QLoRA** = **Q**uantized **Lo**w-**R**ank **A**daptation.

The base model is kept in 4-bit precision (frozen — weights do not change). Small trainable "adapter" matrices (LoRA) are added to the attention and feed-forward layers. Only the adapters (~0.5% of parameters) are trained.

```
Full fine-tune memory: ~16 GB (BF16 weights + optimizer states)
QLoRA memory:          ~4 GB  (4-bit base + BF16 adapter + optimizer)
```

This makes fine-tuning feasible on DataHub A100s.

### Training Dataset: MATH

**Source**: `lighteval/MATH` on HuggingFace (7,500 competition math problems with step-by-step solutions).

The dataset spans the same domains as the competition: algebra, number theory, counting & probability, geometry, precalculus, intermediate algebra. Solutions end naturally with `\boxed{answer}`.

**Training format** (one example):
```
<|im_start|>system
You are an expert mathematician. Solve the problem step-by-step...
<|im_end|>
<|im_start|>user
Let f(x) = 4x^2 + x + 2, find f(3).
<|im_end|>
<|im_start|>assistant
We substitute x = 3 into f(x):
f(3) = 4(3)^2 + 3 + 2 = 36 + 3 + 2 = 41
\boxed{41}
<|im_end|>
```

The model learns to produce detailed, well-structured reasoning that ends with `\boxed{}`.

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model quantization | 4-bit NF4 | Minimizes VRAM while preserving accuracy |
| LoRA rank (r) | 16 | Balance between expressivity and parameter count |
| LoRA alpha | 32 | Scaling factor; α = 2r is standard |
| LoRA dropout | 0.05 | Light regularization |
| LoRA target modules | q, k, v, o projections + gate/up/down FFN | All attention + FFN layers |
| Optimizer | paged_adamw_8bit | 8-bit optimizer, reduces VRAM |
| Learning rate | 2×10⁻⁴ | Standard for LoRA fine-tuning |
| LR schedule | Cosine with 3% warmup | Smooth decay |
| Batch size | 1 per GPU | Constrained by sequence length |
| Gradient accumulation | 8 | Effective batch = 8 |
| Epochs | 3 | Standard for SFT on 7,500 examples |
| Max sequence length | 4,096 tokens | Balances coverage vs. memory |

### Trainable parameters

LoRA adds adapters to 7 module types across all transformer layers (~28 layers × 7 = ~196 adapter pairs). With rank=16, each adapter has 2 matrices of shape `(hidden_dim, 16)` and `(16, hidden_dim)`. Total trainable: ~17M parameters out of ~4B = **~0.4% of model**.

### Training output

```
checkpoints/model3_qlora/
├── adapter_config.json       # LoRA configuration (rank, target modules, etc.)
├── adapter_model.safetensors # Trained adapter weights
├── tokenizer.json            # Tokenizer (copied from base model)
├── training_log.jsonl        # {"step": 10, "loss": 1.42}, {"step": 20, "loss": 1.31}, ...
└── checkpoint-*/             # Intermediate checkpoints (one per epoch)
```

### Running — Training

```bash
# All dependencies including peft/trl/datasets are in requirements.txt
pip install -r requirements.txt -q

# Quick smoke test (20 steps, 100 examples)
python model3_finetune_train.py --max_steps 20 --subset 100

# Full training (3 epochs, ~7,500 examples, ~6 hours on A100)
python model3_finetune_train.py --epochs 3 --gpu 0

# On DataHub: set HF cache to scratch to avoid home-dir quota
python model3_finetune_train.py --hf_cache /scratch/$USER/hf_cache
```

### Running — Inference

```bash
# Evaluate on public set (with scoring)
python model3_finetune_infer.py --checkpoint checkpoints/model3_qlora

# Generate Kaggle submission from private test set (no answers available)
python model3_finetune_infer.py --checkpoint checkpoints/model3_qlora \
    --data data/private.jsonl --no_eval

# Limit for testing
python model3_finetune_infer.py --checkpoint checkpoints/model3_qlora --limit 50
```

### Output files

```
results/model3_finetune_results.jsonl   # Per-question scores (public set)
results/model3_submission.csv           # Kaggle submission format
```

---

## 9. Reading Result Files

All result files are in `results/` as newline-delimited JSON (`.jsonl`). Each line is one question.

### Loading results in Python

```python
import json

with open("results/model1_v1_enhanced_cot_results.jsonl") as f:
    results = [json.loads(line) for line in f]

mcq_results  = [r for r in results if r["is_mcq"]]
free_results = [r for r in results if not r["is_mcq"]]

overall_acc  = sum(r["correct"] for r in results) / len(results) * 100
mcq_acc      = sum(r["correct"] for r in mcq_results) / len(mcq_results) * 100
free_acc     = sum(r["correct"] for r in free_results) / len(free_results) * 100

print(f"Overall: {overall_acc:.2f}%  MCQ: {mcq_acc:.2f}%  Free-form: {free_acc:.2f}%")
```

### Plotting training loss (Model 3)

```python
import json
import matplotlib.pyplot as plt

log = [json.loads(l) for l in open("checkpoints/model3_qlora/training_log.jsonl")]
steps = [r["step"] for r in log]
losses = [r["loss"] for r in log]

plt.figure(figsize=(8, 4))
plt.plot(steps, losses)
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("Model 3 QLoRA Training Loss")
plt.savefig("loss_curve.pdf")
```

### Comparing models

```python
import json

def load_acc(path):
    results = [json.loads(l) for l in open(path)]
    return {
        "overall": sum(r["correct"] for r in results) / len(results) * 100,
        "mcq":     sum(r["correct"] for r in results if r["is_mcq"]) / sum(1 for r in results if r["is_mcq"]) * 100,
        "free":    sum(r["correct"] for r in results if not r["is_mcq"]) / sum(1 for r in results if not r["is_mcq"]) * 100,
    }

models = {
    "Baseline (v0)":          "results/model1_v0_baseline_results.jsonl",
    "Prompt Eng. (v1)":       "results/model1_v1_enhanced_cot_results.jsonl",
    "Few-shot (v2)":          "results/model1_v2_fewshot_results.jsonl",
    "Verification (v3)":      "results/model1_v3_verification_results.jsonl",
    "Voting N=5":             "results/model2_voting_n5_results.jsonl",
    "Fine-tuned (QLoRA)":     "results/model3_finetune_results.jsonl",
}

for name, path in models.items():
    acc = load_acc(path)
    print(f"{name:<25}  Overall={acc['overall']:.2f}%  MCQ={acc['mcq']:.2f}%  Free={acc['free']:.2f}%")
```

---

## 10. Running on DataHub

Open a terminal in JupyterLab (**File → New → Terminal**) and run everything from there.

### Pre-flight checks

```bash
# 1. Confirm you are on a GPU node
nvidia-smi
# Expected: table showing an A100 (or similar). If not, request a GPU node.

# 2. Navigate to the project folder
cd ~/CSE151B/Project          # adjust to wherever you uploaded the files
ls                            # should list model1_*.py, model2_*.py, model3_*.py, data/, etc.

# 3. Install all dependencies from requirements.txt
#    DataHub already has a CUDA-enabled torch, so pip resolves everything else.
pip install -r requirements.txt -q

# 4. Confirm torch sees the GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.x.x  True

# 5. Confirm private.jsonl is present (needed for Kaggle submission)
ls data/
# Expected: public.jsonl  private.jsonl
# If private.jsonl is missing: download it from Kaggle → Data tab and upload to data/

# 6. Create output directories
mkdir -p logs results checkpoints/model3_qlora
```

> **Why not `.venv`?** The `.venv` folder is not committed to git and won't exist after a fresh clone. `requirements.txt` is the portable alternative — it installs into whatever Python environment DataHub provides.

---

### Resume behaviour

All three inference scripts (**model1**, **model2**, **model3_finetune_infer**) write each result to disk immediately after it is generated. If a DataHub session is killed mid-run:

- The output `.jsonl` file already contains every question processed so far.
- Re-running the **exact same command** skips those questions and picks up from where it stopped.
- No data is lost and no time is wasted re-running completed questions.

The training script (**model3_finetune_train**) saves a full checkpoint at the end of each epoch via HuggingFace `TrainingArguments(save_strategy="epoch")`. If training is killed inside an epoch the current epoch must be re-run, but completed epochs are safe.

---

### Time budget

| Step | What runs | Output | Est. time |
|------|-----------|--------|-----------|
| Smoke tests | All 4 scripts, 5 questions each | Confirms scripts work | ~2 min |
| Model 1 | All 4 prompt variants, 1126 questions | 4 × JSONL result files | ~50 min |
| Model 2 — temp sweep | 5 temperatures × 1126 questions | 5 × JSONL result files | ~60 min |
| Model 2 — voting N=5 | 5 samples × 1126 questions | 1 × JSONL result file | ~60 min |
| Model 3 training | 3 epochs × 7500 MATH examples | LoRA adapter checkpoint | ~6 hours |
| Model 3 inference | 1126 questions with fine-tuned model | JSONL + submission CSV | ~12 min |

**Strategy**: kick off Model 3 training first (runs overnight in the background), then run Models 1 and 2 while it trains. Everything finishes within a single DataHub session.

---

### Step-by-step commands

#### Step 0 — Smoke tests (~2 min)

Always run these before committing GPU hours to a full run.

```bash
python model1_prompt_engineering.py --variant v0_baseline --limit 5
# Expected: 5 responses printed + a small accuracy table

python model2_sampling_voting.py --experiment voting_n3 --limit 5
# Expected: 3 samples generated for each of 5 questions + accuracy table

python model3_finetune_train.py --max_steps 5 --subset 50
# Expected: downloads MATH dataset, prints training config, runs 5 steps, saves adapter

python model3_finetune_infer.py --checkpoint checkpoints/model3_qlora --limit 5
# Expected: loads adapter, generates 5 responses, prints accuracy
```

If any step fails, fix the error before proceeding (usually a missing package or wrong working directory).

---

#### Step 1 — Start Model 3 training in the background (~6 hours)

Do this **first** so it runs while you work on everything else.

```bash
# Set HF model/dataset cache to scratch space to avoid home-directory quota
export HF_HOME=/datasets/$USER/hf_cache    # adjust path for your DataHub setup
mkdir -p $HF_HOME logs

nohup python model3_finetune_train.py \
    --epochs 3 --gpu 0 --hf_cache $HF_HOME \
    > logs/model3_train.log 2>&1 &

echo "Training PID: $!"
```

Monitor while it runs:

```bash
tail -f logs/model3_train.log                    # live log stream (Ctrl+C to stop watching)
tail -5 checkpoints/model3_qlora/training_log.jsonl  # latest loss values
ps aux | grep model3_finetune_train              # confirm the process is still alive
```

Training is complete when `checkpoints/model3_qlora/adapter_model.safetensors` exists.

> **Disk quota error?** The MATH dataset + model weights need ~10 GB. Use `--hf_cache /scratch/$USER/hf_cache` or another large-quota path.

---

#### Step 2 — Model 1: Prompt Engineering (~50 min)

```bash
nohup python model1_prompt_engineering.py --variant all --gpu 0 \
    > logs/model1.log 2>&1 &

echo "Model 1 PID: $!"
tail -f logs/model1.log
```

Check partial results at any point (safe to run while the script is still going):

```bash
python3 - <<'EOF'
import json, os
variants = ["v0_baseline","v1_enhanced_cot","v2_fewshot","v3_verification"]
print(f"\n{'Variant':<25} {'MCQ':>8} {'Free':>8} {'Overall':>10} {'N':>6}")
print("-"*62)
for v in variants:
    path = f"results/model1_{v}_results.jsonl"
    if not os.path.exists(path):
        print(f"{v:<25}  — not started"); continue
    rs = [json.loads(l) for l in open(path)]
    m = [r for r in rs if r["is_mcq"]]; f = [r for r in rs if not r["is_mcq"]]
    p = lambda s: sum(r["correct"] for r in s)/len(s)*100 if s else 0
    print(f"{v:<25} {p(m):>7.2f}% {p(f):>7.2f}% {p(rs):>9.2f}% {len(rs):>6}")
EOF
```

Output files:
```
results/model1_v0_baseline_results.jsonl
results/model1_v1_enhanced_cot_results.jsonl
results/model1_v2_fewshot_results.jsonl
results/model1_v3_verification_results.jsonl
```

---

#### Step 3 — Model 2: Temperature Sweep (~60 min)

```bash
nohup python model2_sampling_voting.py --experiment temp_sweep --gpu 0 \
    > logs/model2_temp.log 2>&1 &

echo "Temp sweep PID: $!"
tail -f logs/model2_temp.log
```

Output files:
```
results/model2_temp0p0_results.jsonl   ← greedy (T=0, deterministic)
results/model2_temp0p3_results.jsonl
results/model2_temp0p5_results.jsonl
results/model2_temp0p7_results.jsonl
results/model2_temp0p9_results.jsonl
```

---

#### Step 4 — Model 2: Majority Voting (~60 min for N=5)

```bash
nohup python model2_sampling_voting.py --experiment voting_n5 --gpu 0 \
    > logs/model2_vote5.log 2>&1 &

echo "Voting N=5 PID: $!"
```

Optional — run N=3 and N=7 as well if time and GPU budget allow:

```bash
nohup python model2_sampling_voting.py --experiment voting_n3 --gpu 0 > logs/model2_vote3.log 2>&1 &
nohup python model2_sampling_voting.py --experiment voting_n7 --gpu 0 > logs/model2_vote7.log 2>&1 &
```

Check partial results:

```bash
python3 - <<'EOF'
import json, glob, os
files = sorted(glob.glob("results/model2_*.jsonl"))
print(f"\n{'Experiment':<25} {'MCQ':>8} {'Free':>8} {'Overall':>10} {'N':>6}")
print("-"*62)
for path in files:
    rs = [json.loads(l) for l in open(path)]
    label = path.split("/")[-1].replace("model2_","").replace("_results.jsonl","")
    m = [r for r in rs if r["is_mcq"]]; f = [r for r in rs if not r["is_mcq"]]
    p = lambda s: sum(r["correct"] for r in s)/len(s)*100 if s else 0
    print(f"{label:<25} {p(m):>7.2f}% {p(f):>7.2f}% {p(rs):>9.2f}% {len(rs):>6}")
EOF
```

---

#### Step 5 — Model 3 Inference (~12 min, after training finishes)

```bash
# Confirm training is done
ls checkpoints/model3_qlora/adapter_model.safetensors

# Run inference on public set (scored)
nohup python model3_finetune_infer.py \
    --checkpoint checkpoints/model3_qlora --gpu 0 \
    > logs/model3_infer.log 2>&1 &

tail -f logs/model3_infer.log
```

Output files:
```
results/model3_finetune_results.jsonl
results/model3_submission.csv
```

---

### Getting all numbers at once

Run this after all experiments finish to print the complete comparison table for the report:

```bash
python3 - <<'EOF'
import json, os

def load(path):
    if not os.path.exists(path): return None
    rs = [json.loads(l) for l in open(path)]
    m = [r for r in rs if r["is_mcq"]]; f = [r for r in rs if not r["is_mcq"]]
    p = lambda s: sum(r["correct"] for r in s)/len(s)*100 if s else 0
    return p(m), p(f), p(rs), len(rs)

rows = [
    ("Baseline (v0)",             "results/model1_v0_baseline_results.jsonl"),
    ("Prompt: Enhanced CoT (v1)", "results/model1_v1_enhanced_cot_results.jsonl"),
    ("Prompt: Few-shot (v2)",     "results/model1_v2_fewshot_results.jsonl"),
    ("Prompt: Verification (v3)", "results/model1_v3_verification_results.jsonl"),
    ("Temp: Greedy (T=0.0)",      "results/model2_temp0p0_results.jsonl"),
    ("Temp: T=0.3",               "results/model2_temp0p3_results.jsonl"),
    ("Temp: T=0.5",               "results/model2_temp0p5_results.jsonl"),
    ("Temp: T=0.7",               "results/model2_temp0p7_results.jsonl"),
    ("Temp: T=0.9",               "results/model2_temp0p9_results.jsonl"),
    ("Voting N=3",                "results/model2_voting_n3_results.jsonl"),
    ("Voting N=5",                "results/model2_voting_n5_results.jsonl"),
    ("Voting N=7",                "results/model2_voting_n7_results.jsonl"),
    ("Fine-tuned QLoRA",          "results/model3_finetune_results.jsonl"),
]

print(f"\n{'Model':<30} {'MCQ':>8} {'Free':>8} {'Overall':>10}  N")
print("="*65)
for name, path in rows:
    r = load(path)
    if r: print(f"{name:<30} {r[0]:>7.2f}% {r[1]:>7.2f}% {r[2]:>9.2f}%  {r[3]}")
    else: print(f"{name:<30}  — not run yet")
print()
EOF
```

You can run this script at any point — rows for experiments that haven't finished yet will show "— not run yet".

---

### Background job management

```bash
# List all running Python scripts and their PIDs
ps aux | grep python | grep -v grep

# Watch a specific log file live
tail -f logs/model1.log

# See all log files and when they were last updated
ls -lth logs/

# Kill a job if needed (replace PID with the number from ps aux)
kill <PID>
```

---

### Troubleshooting

| Error | Fix |
|-------|-----|
| `nvidia-smi: command not found` | You are on a CPU-only node — request a GPU node from DataHub |
| `CUDA out of memory` | Try `--gpu 1` to use a different GPU; check VRAM with `nvidia-smi` |
| `No module named 'trl'` | Run `pip install -r requirements.txt -q` |
| `FileNotFoundError: data/public.jsonl` | Run `cd ~/CSE151B/Project` to make sure you are in the project root |
| HuggingFace download hangs or fails | Set `--hf_cache /datasets/$USER/hf_cache` (larger quota than home dir) |
| Training loss is NaN from step 1 | Reduce learning rate with `--lr 1e-4`; check `training_log.jsonl` |
| Script killed mid-run | Re-run the **exact same command** — completed questions are already saved and will be skipped |
| Results file has fewer than 1126 lines | Script was interrupted; re-run to resume from the last completed question |

---

## 11. Milestone Report Writing Guide

The LaTeX source is at `Milestone Report/latex/main.tex`. Below is a section-by-section guide for what to write and which numbers to pull from the results.

---

### Abstract

Include:
- One sentence on the competition task (math reasoning, Qwen3-4B-Thinking-2507)
- The three approaches explored (prompt engineering, sampling/voting, fine-tuning)
- Best result achieved (pull `overall_acc` from whichever model performed best)
- Kaggle leaderboard rank/score if available

---

### Section 1: Introduction

**Problem Definition**  
The task is to improve the accuracy of a 4B-parameter language model on a diverse set of mathematical reasoning problems, spanning high-school to graduate level, without using external tools at inference time.

**Problem Significance**  
Mathematical reasoning is a key capability for scientific automation, quantitative analysis, and educational applications. Small efficient models (4B params) that can reason accurately are especially valuable for deployment without large compute.

**Technical Challenges**
- *Data*: Problems span many domains (algebra, calculus, statistics) and formats (MCQ, free-form, multi-part).
- *Model*: The 4B model is smaller and has less capacity than 70B+ models that dominate benchmarks.
- *Evaluation*: Free-form symbolic answers require fuzzy matching (SymPy equivalence), not exact string match.
- *Compute*: Fine-tuning and majority voting are GPU-intensive; the starter notebook uses 4-bit quantization to fit on limited hardware.

**Contributions** — enumerate the 3 models:
1. Prompt Engineering: 4 variants compared
2. Sampling & Majority Voting: temperature sweep + N-sample self-consistency
3. QLoRA Fine-Tuning: SFT on 7,500 competition math problems

---

### Section 2: Related Work (Optional)

Suggested references (add to `references.bib`):
- **Chain-of-Thought Prompting**: Wei et al. 2022 — introduced CoT; explains why step-by-step prompts help.
- **Self-Consistency**: Wang et al. 2023 — original majority voting paper for LLMs.
- **LoRA**: Hu et al. 2022 — low-rank adaptation of large language models.
- **QLoRA**: Dettmers et al. 2023 — quantized LoRA for memory-efficient fine-tuning.
- **MATH dataset**: Hendrycks et al. 2021 — the training dataset used for fine-tuning.

---

### Section 3: Methods

**Problem Setting**  
Input: a math problem in LaTeX. Output: one or more answers in `\boxed{}`.  
Evaluation: unified accuracy (correct/total, each question weighted equally).

**Baseline** — describe the starter notebook:
> The baseline runs Qwen3-4B-Thinking-2507 with INT4 quantization (BitsAndBytes NF4), generic system prompts, and sampling parameters `T=0.6, top_p=0.95, top_k=20`.

**Model 1 — Prompt Engineering**: *(~2 paragraphs)*
> We test 4 prompt variants while holding all other factors constant. v0 is the baseline. v1 adds explicit numbered CoT steps. v2 prepends 2 worked examples per question type. v3 instructs the model to verify its answer before committing.

**Model 2 — Sampling & Voting**: *(~2 paragraphs)*
> We sweep temperature from 0.0 (greedy) to 0.9 to identify the optimal single-sample regime. We then apply self-consistency [cite Wang 2023] with N=3, 5, 7 samples at T=0.7: multiple independent responses are generated and the modal answer is selected.

**Model 3 — QLoRA Fine-Tuning**: *(~3 paragraphs)*
> We fine-tune using QLoRA [cite Dettmers 2023] on the MATH dataset [cite Hendrycks 2021] (7,500 competition math problems). The base model is frozen in 4-bit; LoRA adapters (rank 16, alpha 32) are trained on all attention projection and FFN layers. Training uses paged AdamW with `lr=2×10⁻⁴`, cosine schedule, effective batch size 8, for 3 epochs.

---

### Section 4: Experiments

**Baselines subsection** — list the 4 models:
- Starter code (v0 baseline)
- Prompt engineering best variant
- Majority voting best N
- Fine-tuned QLoRA

**Evaluation subsection**  
Local evaluation on `public.jsonl` (1,126 questions). The judger handles symbolic equivalence. MCQ accuracy and free-form accuracy reported separately; unified accuracy is the primary metric (matches Kaggle scoring).

**Implementation Details table** — fill from actual runs:

| | Baseline | Model 1 | Model 2 | Model 3 |
|---|---|---|---|---|
| GPU | A100 | A100 | A100 | A100 |
| Quantization | INT4 | INT4 | INT4 | INT4 base + BF16 adapter |
| Temperature | 0.6 | 0.6 | 0.7 (voting) | 0.6 |
| Samples/question | 1 | 1 | N=5 | 1 |
| Fine-tuning data | — | — | — | MATH (7,500) |
| LoRA rank | — | — | — | 16 |
| Epochs | — | — | — | 3 |
| Runtime | ~12 min | ~12 min/variant | ~60 min | ~6 hr train + ~12 min infer |

**Results table** — fill in XX.XX with actual numbers from the JSONL files:

```latex
\begin{table}[h]
\caption{Accuracy on the public evaluation set (1,126 questions).}
\label{tab:results}
\centering
\begin{tabular}{lccc}
\toprule
Model & MCQ & Free-form & Overall \\
\midrule
Baseline (starter code)      & XX.XX\% & XX.XX\% & XX.XX\% \\
Prompt Eng. — v1 Enhanced CoT & XX.XX\% & XX.XX\% & XX.XX\% \\
Prompt Eng. — v2 Few-shot     & XX.XX\% & XX.XX\% & XX.XX\% \\
Prompt Eng. — v3 Verification & XX.XX\% & XX.XX\% & XX.XX\% \\
\midrule
Sampling: Greedy (T=0.0)      & XX.XX\% & XX.XX\% & XX.XX\% \\
Sampling: T=0.7               & XX.XX\% & XX.XX\% & XX.XX\% \\
Majority Voting N=3           & XX.XX\% & XX.XX\% & XX.XX\% \\
Majority Voting N=5           & XX.XX\% & XX.XX\% & XX.XX\% \\
Majority Voting N=7           & XX.XX\% & XX.XX\% & XX.XX\% \\
\midrule
QLoRA Fine-tuned              & XX.XX\% & XX.XX\% & XX.XX\% \\
\bottomrule
\end{tabular}
\end{table}
```

**Qualitative result suggestion**: Pick one example question where the baseline fails but a better model succeeds. Show the two responses (condensed) side by side.

**Training loss curve (Model 3)**: Generate a plot from `checkpoints/model3_qlora/training_log.jsonl` (see snippet in Section 9) and include as a figure.

---

### Section 5: Discussion

**Achievements** — fill in with actual numbers:
- Best prompt variant improved overall accuracy from XX% → XX% (ΔXX pp)
- Majority voting N=5 improved over single-sample by ΔXX pp
- Fine-tuning improved by ΔXX pp over baseline

**Bottlenecks**:
- GPU memory: 4-bit quantization was necessary to fit on DataHub hardware
- Majority voting runtime: N=7 requires 7× inference time
- Fine-tuning time: ~6 hours for 3 epochs limits hyperparameter search

**Plans Forward**:
- Combine best prompt with majority voting (orthogonal improvements)
- Try GRPO (reinforcement learning) after SFT warm-up
- Increase LoRA rank or epochs if compute allows

---

## 12. Generating a Kaggle Submission

The private test set (`data/private.jsonl`) has no answers — submit predicted responses to Kaggle.

### Using the fine-tuned model

```bash
python model3_finetune_infer.py \
    --checkpoint checkpoints/model3_qlora \
    --data data/private.jsonl \
    --no_eval \
    --output results/
```

Output: `results/model3_private_submission.csv`

### Submission format

```csv
id,response
0,"<think>...reasoning...</think>\n\nThe answer is \boxed{42}"
1,"<think>...reasoning...</think>\n\nThe answer is \boxed{580, 660, 80}"
```

The response column must contain the **full model output** (including chain-of-thought). The Kaggle judger extracts `\boxed{}` from the response during evaluation.

### Manually generating a submission from any JSONL result file

```python
import csv, json

results = [json.loads(l) for l in open("results/model2_voting_n5_results.jsonl")]

with open("results/voting_n5_submission.csv", "w", newline="") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(["id", "response"])
    for r in results:
        # For voting: use the first sample as the response (contains the voted reasoning)
        response = r["responses"][0] if isinstance(r.get("responses"), list) else r["response"]
        writer.writerow([r["id"], response])
```

> **Note**: The `voted` field in model2 results is the extracted normalized answer, not the full response. For submission, use `responses[0]` (the first sample) — it contains the full reasoning trace and the judger will extract the answer automatically.

---

*Last updated: May 2026 · Vishudh Shah*
