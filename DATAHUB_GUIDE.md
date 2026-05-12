# DataHub Guide — CSE 151B Competition

Everything you need to run experiments on UCSD DSMLP from scratch.

> **Official UCSD docs**: [KB0032269 — DSMLP getting started guide](https://support.ucsd.edu/services?id=kb_article_view&sysparm_article=KB0032269) — covers SSH setup, `launch.sh` options, and general DSMLP usage.

---

## Table of Contents

1. [Connect to the login node](#1-connect-to-the-login-node)
2. [Launch a GPU node](#2-launch-a-gpu-node)
3. [Open JupyterLab](#3-open-jupyterlab)
4. [Clone the repo and set up git credentials](#4-clone-the-repo-and-set-up-git-credentials)
5. [Install dependencies](#5-install-dependencies)
6. [Keep jobs alive with tmux](#6-keep-jobs-alive-with-tmux)
7. [Run the experiments](#7-run-the-experiments)
8. [Check results mid-run](#8-check-results-mid-run)
9. [Commit results](#9-commit-results)
10. [Resume after a killed session](#10-resume-after-a-killed-session)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Connect to the login node

From your laptop (must be on UCSD VPN or on campus):

```bash
ssh <username>@dsmlp-login.ucsd.edu
```

You land on the login node — this is a CPU-only machine just for launching jobs. Do not run experiments here.

---

## 2. Launch a GPU node

Use the course-specific script (instructor-configured image with CUDA 12.8, additional GPU pool). Pods are killed after 6 hours by default — set `K8S_TIMEOUT_SECONDS` first to extend to 12 hours (required for Model 3 training):

```bash
export K8S_TIMEOUT_SECONDS=43200
launch-sp26-cuda128.sh -l gpu-class=medium -W CSE151B_SP26_A00 -g 1 -c 16 -m 64
```

- `-l gpu-class=medium` — Kubernetes label selector; targets the GPU pool provisioned for this course
- `-W CSE151B_SP26_A00` — mounts the course-specific workspace directory and applies the course resource quota
- `-g 1` — 1 GPU
- `-c 16` — 16 CPUs
- `-m 64` — 64 GB RAM

If this hangs on "Pending" for more than ~30 seconds, fall back to the general script:

```bash
# Try GPU types in order of availability: a30, a5000, a100, h100
launch.sh -v a30 -g 1 -c 16 -m 64 -W CSE151B_SP26_A00
```

Add `-s` to either command if you want to skip Jupyter and use only the terminal (saves ~10 seconds on startup and avoids the browser step):

```bash
launch-sp26-cuda128.sh -l gpu-class=medium -W CSE151B_SP26_A00 -g 1 -c 16 -m 64 -s
```

When it starts, you'll see a URL like:
```
You may access your Jupyter notebook at: http://dsmlp-login:XXXXX/user/<username>/?token=...
```

---

## 3. Open JupyterLab

Open that URL in your browser (**must be on VPN**). Then open a terminal: **File → New → Terminal**.

Verify you have a GPU:
```bash
nvidia-smi
# GPU varies by node — e.g. RTX PRO 6000 Blackwell (MIG slice, 24 GB) or A30 (24 GB)
# CUDA Version: 13.0
```

---

## 4. Clone the repo and set up git credentials

If this is a fresh node (no project files yet):
```bash
cd ~
git clone https://github.com/vishudhshah/cse151b-project.git "private/CSE 151B/cse151b-project"
cd "private/CSE 151B/cse151b-project"
```

If the project folder already exists:
```bash
cd ~/private/CSE\ 151B/cse151b-project
git pull
```

### Set up git credentials (needed every new pod)

Each DataHub pod is a fresh machine with no SSH keys. Use HTTPS with a stored token:

```bash
git config --global credential.helper store
git remote set-url origin https://github.com/vishudhshah/cse151b-project.git
```

The first `git push` will prompt for your GitHub username and a personal access token (PAT). Generate one at:
**GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)**
— scope: `repo`. After entering it once, it's stored for the session.

---

## 5. Install dependencies

**Returning to the same pod?** The venv persists — skip this section entirely and just run:

```bash
source ~/private/CSE\ 151B/cse151b-project/.venv/bin/activate
```

**Got a new pod?** Run the setup below (~1 min — torch and most packages are pre-installed in the image):

```bash
cd ~/private/CSE\ 151B/cse151b-project

# Create venv that inherits pre-installed packages (torch 2.11+cu128, transformers, accelerate, sympy, etc.)
uv venv .venv --seed --system-site-packages
source .venv/bin/activate

# Install from requirements.txt — use pip (not uv) so system-site-packages deps are skipped
pip install -r requirements.txt

# Set HuggingFace token to avoid rate limiting (get from huggingface.co/settings/tokens)
export HF_TOKEN=hf_your_token_here

# Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.11.0+cu128  True
```

---

## 6. Keep jobs alive with tmux

DataHub pods disconnect if your browser/VPN drops. Use `tmux` so jobs keep running:

```bash
tmux new -s experiments      # start a named session
# ... run your commands inside ...
# Ctrl+B then D              — detach (leaves session running)
tmux attach -t experiments   # reattach later
tmux ls                      # list all sessions
```

Always start a tmux session before launching long-running experiments.

---

## 7. Run the experiments

### Smoke tests first (~5 min)

Confirm everything works before committing hours of GPU time:

```bash
mkdir -p logs results checkpoints/model3_qlora

python model1_prompt_engineering.py --variant v0_baseline --limit 5 --max_tokens 2048
python model2_sampling_voting.py --experiment voting_n3 --limit 5 --max_tokens 2048
python model3_finetune_train.py --max_steps 5 --subset 50
python model3_finetune_infer.py --checkpoint checkpoints/model3_qlora --limit 5 --max_tokens 2048
```

> **Progress bar looks stuck on first question?** Normal — the thinking model generates up to 4,096 tokens before answering. The bar only ticks after each complete response. `--max_tokens 1024` makes smoke tests faster; use the default for real runs.

### Full experiment run

Start Model 3 training first (~10 hours) and let it run alone — running inference jobs in parallel causes GPU contention and blows out ETAs.

```bash
# ── Step 1: Model 3 training (start first, let it finish before steps 2–4) ───
nohup python model3_finetune_train.py --epochs 3 > logs/model3_train.log 2>&1 &
echo "Training PID: $!"

# ── After training finishes: run inference immediately while checkpoint is fresh
ls checkpoints/model3_qlora/adapter_model.safetensors  # confirm training done
nohup python model3_finetune_infer.py --checkpoint checkpoints/model3_qlora > logs/model3_infer.log 2>&1 &

# ── Step 2: Model 1 — baseline variant only ───────────────────────────────────
nohup python model1_prompt_engineering.py --variant v0_baseline > logs/model1.log 2>&1 &
echo "Model 1 PID: $!"

# ── Step 3: Model 2 — majority voting N=3 (fastest experiment) ────────────────
nohup python model2_sampling_voting.py --experiment voting_n3 > logs/model2_vote3.log 2>&1 &
```

> **If Model 3 inference crashes with `torch.cuda.OutOfMemoryError`:** the GPU ran out of VRAM from holding two questions in memory simultaneously. Re-run with `--batch_size 1` — it processes one question at a time and resumes from where it left off.

Monitor any job:
```bash
tail -f logs/model1.log              # live output
ps aux | grep python | grep -v grep  # see all running jobs
kubectl get pod                      # list your pods (from the login node)
kubectl logs <pod-name>              # get output from a background pod
kubectl delete pod <pod-name>        # manually terminate a pod when done
```

> Check GPU availability before launching: [datahub.ucsd.edu/hub/status](https://datahub.ucsd.edu/hub/status) (requires UCSD login).

### Time estimates (A30 GPU)

| Step | Est. time |
|------|-----------|
| Model 3 — training (3 epochs) | ~10 hours |
| Model 1 — baseline variant | ~30 min |
| Model 2 — voting N=3 | ~20 min |
| Model 3 — inference | ~12 min |

---

## 8. Check results mid-run

Print accuracy for any experiments that have finished so far:

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
    ("Baseline (v0)",    "results/model1_v0_baseline_results.jsonl"),
    ("Voting N=3",       "results/model2_voting_n3_results.jsonl"),
    ("Fine-tuned QLoRA", "results/model3_finetune_results.jsonl"),
]

print(f"\n{'Model':<30} {'MCQ':>8} {'Free':>8} {'Overall':>10}  N")
print("="*65)
for name, path in rows:
    r = load(path)
    if r: print(f"{name:<30} {r[0]:>7.2f}% {r[1]:>7.2f}% {r[2]:>9.2f}%  {r[3]}")
    else: print(f"{name:<30}  — not run yet")
EOF
```

---

## 9. Commit results

After experiments finish, push results and checkpoint configs to the repo so teammates can access them:

```bash
git add results/ checkpoints/
git commit -m "Add experiment results"
git push
```

The large weight files (`*.safetensors`, `*.pt`, `*.bin`) are gitignored automatically — only the small config/JSON files get committed.

---

## 10. Resume after a killed session

All inference scripts write results one question at a time. If a session dies mid-run:

1. Launch a new pod: `launch-sp26-cuda128.sh -l gpu-class=medium -W CSE151B_SP26_A00 -g 1 -c 16 -m 64`
2. `cd` into the project, `source .venv/bin/activate`
3. Re-run the **exact same command** — it detects already-completed questions and skips them

For training, completed epochs are checkpointed. If killed mid-epoch, re-run from the last saved checkpoint:
```bash
# Resume from the last epoch checkpoint
python model3_finetune_train.py --epochs 3 --output checkpoints/model3_qlora  # trainer resumes automatically
```

---

## 11. Troubleshooting

| Error | Fix |
|-------|-----|
| `launch-sp26-cuda128.sh` hangs on Pending | Use fallback: `launch.sh -v a30 -g 1 -c 16 -m 64` |
| `source .venv/bin/activate` fails | Venv doesn't exist on this pod — run the install from Section 5 |
| `torch.cuda.is_available()` is False | Check `nvidia-smi` first; if driver is present, recreate venv: `rm -rf .venv` then re-run Section 5 |
| `git@github.com: Permission denied` | Node has no SSH key — switch to HTTPS: `git remote set-url origin https://github.com/vishudhshah/cse151b-project.git` |
| `No module named 'trl'` | Venv not activated — run `source .venv/bin/activate` |
| `FileNotFoundError: results/...` | `results/` directory missing — run `mkdir -p results logs checkpoints/model3_qlora` |
| Progress bar stuck | Normal for first question — see note in Section 7 |
| HuggingFace download slow | Set `export HF_TOKEN=hf_...` (token from huggingface.co/settings/tokens) |
| Disk quota error during training | Model weights cache is large — `export HF_HOME=/datasets/$USER/hf_cache` |
| Pod status: `OOMKilled` | Ran out of RAM — relaunch with `-m 48` or `-m 64` |
| Pod status: `DeadlineExceeded` | Hit the 6-hour pod limit — set `export K8S_TIMEOUT_SECONDS=43200` before launching (see Section 2) |
