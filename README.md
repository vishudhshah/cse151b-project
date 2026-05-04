# CSE 151B Spring 2026 — Math Reasoning Competition

**Model**: Qwen/Qwen3-4B-Thinking-2507 &nbsp;|&nbsp; **Team**: Rohan Raval, Tanishq Rathore, Vishudh Shah

> **Writing the milestone report or running experiments on DataHub?** Read [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) — it covers all models, how to run them, resume behaviour if jobs are killed, and a section-by-section guide for the report.

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
├── PROJECT_GUIDE.md              # ← Full documentation for teammates
└── README.md                     # This file
```

## Quick Commands

```bash
# Baseline (interactive, run in Jupyter)
jupyter notebook starter_code_cse151b_comp.ipynb

# Model 1 — smoke test
python model1_prompt_engineering.py --variant v0_baseline --limit 20

# Model 1 — full run, all variants
python model1_prompt_engineering.py --variant all

# Model 2 — temperature sweep
python model2_sampling_voting.py --experiment temp_sweep

# Model 2 — majority voting (5 samples)
python model2_sampling_voting.py --experiment voting_n5

# Model 3 — fine-tune (needs: pip install trl peft datasets)
python model3_finetune_train.py --epochs 3

# Model 3 — inference with fine-tuned model
python model3_finetune_infer.py --checkpoint checkpoints/model3_qlora
```

See [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) for full details.
