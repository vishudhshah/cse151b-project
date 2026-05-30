# CSE 151B Spring 2026 — Math Reasoning Competition

**Model**: Qwen/Qwen3-4B-Thinking-2507 &nbsp;|&nbsp; **Team**: Rohan Raval, Tanishq Rathore, Vishudh Shah

> **Running experiments?** See [`DATAHUB_GUIDE.md`](DATAHUB_GUIDE.md) — SSH, GPU setup, venv install, tmux, full experiment run.  
> **Writing the report?** See [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) — model descriptions, results format, LaTeX table template.

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
├── model1_prompt_engineering.py  # Model 1: 4 prompt variants compared
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
