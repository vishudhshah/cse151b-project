# CSE 151B Spring 2026 — Math Reasoning Competition

**Model**: Qwen/Qwen3-4B-Thinking-2507 &nbsp;|&nbsp; **Team**: Rohan Raval, Tanishq Rathore, Vishudh Shah

> **Running experiments?** See [`DATAHUB_GUIDE.md`](DATAHUB_GUIDE.md) — SSH, GPU setup, venv install, tmux, full experiment run.  
> **Writing the report?** See [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) — model descriptions, results format, LaTeX table template.

---

## Final Submission

**Strategy**: Model 1 `v2_fewshot` — few-shot examples + enhanced chain-of-thought on the base `Qwen/Qwen3-4B-Thinking-2507` model (no fine-tuning).

### GPU & inference time

GPU: **NVIDIA A30 24 GB**. Approximate inference time on the full private set: **~10 minutes**.

### Model weights

No manual download needed. The base model (`Qwen/Qwen3-4B-Thinking-2507`) is downloaded automatically from HuggingFace Hub on first run. To avoid rate limits, set your token beforehand:

```bash
export HF_TOKEN=hf_your_token_here
```

### Reproducing the submission

```python
from run_inference import run_inference
run_inference()  # reads data/private.jsonl, writes results/model1_submission.csv
```

Or from the command line:

```bash
python run_inference.py
# public set (prints accuracy for verification):
python run_inference.py --data data/public.jsonl
```

---

## Repository Structure

```
.
├── data/
│   └── public.jsonl              # 1126 labeled questions (MCQ + free-form)
│
├── results/                      # Output files written at runtime
│   └── model1_*.jsonl / model2_*.jsonl / model3_*.jsonl
│
├── checkpoints/
│   └── model3_qlora/             # Fine-tuned LoRA adapter weights
│
├── Milestone Report/
│   └── latex/main.tex            # Report LaTeX source
│
├── starter_code_cse151b_comp.ipynb   # Baseline: interactive notebook
├── judger.py                         # Answer-scoring logic (do not modify)
├── utils.py                          # LaTeX / math normalization utilities
│
├── run_inference.py              # ← Final submission entry point (run_inference())
├── model1_prompt_engineering.py  # Model 1: 5 prompt variants compared
├── model2_sampling_voting.py     # Model 2: temperature sweep + majority voting
├── model3_finetune_train.py      # Model 3: QLoRA fine-tuning (training)
├── model3_finetune_infer.py      # Model 3: inference with fine-tuned model
│
├── requirements.txt              # All Python dependencies
├── DATAHUB_GUIDE.md              # ← SSH, GPU setup, running experiments
├── PROJECT_GUIDE.md              # ← Model details, milestone report template
└── README.md                     # This file
```

## Setup

See [`DATAHUB_GUIDE.md`](DATAHUB_GUIDE.md) for the full setup. Quick version:

```bash
uv venv .venv --seed --system-site-packages && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Commands

```bash
# Smoke test — confirm everything works (~2 min)
python model1_prompt_engineering.py --variant v0_baseline --limit 5

# Model 1 — all 4 prompt variants (~40 min)
python model1_prompt_engineering.py --variant all

# Model 2 — temperature sweep (~15 min)
python model2_sampling_voting.py --experiment temp_sweep

# Model 2 — majority voting, 5 samples (~25 min)
python model2_sampling_voting.py --experiment voting_n5

# Model 3 — fine-tune (~10 hours)
python model3_finetune_train.py --epochs 3

# Model 3 — inference with fine-tuned model (~10 min)
python model3_finetune_infer.py --checkpoint checkpoints/model3_qlora
```

See [`DATAHUB_GUIDE.md`](DATAHUB_GUIDE.md) for the full run guide and [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) for model details and the milestone report template.
