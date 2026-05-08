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

Use the course-specific script (instructor-configured image with CUDA 12.8, additional GPU pool):

```bash
launch-sp26-cuda128.sh -l gpu-class=medium -W CSE151B_SP26_A00 -g 1 -c 8 -m 32
```

- `-l gpu-class=medium` — Kubernetes label selector; targets the GPU pool provisioned for this course
- `-W CSE151B_SP26_A00` — mounts the course-specific workspace directory and applies the course resource quota
- `-g 1` — 1 GPU
- `-c 8` — 8 CPUs
- `-m 32` — 32 GB RAM

> Instructor benchmark: 5 responses in 65 seconds on an A30 using this image.

If this hangs on "Pending" for more than ~30 seconds, fall back to the general script:

```bash
# Try GPU types in order of availability: a30, a5000, a100, h100
launch.sh -v a30 -g 1 -c 8 -m 32 -W CSE151B_SP26_A00
```

Add `-s` to either command if you want to skip Jupyter and use only the terminal (saves ~10 seconds on startup and avoids the browser step):

```bash
launch-sp26-cuda128.sh -l gpu-class=medium -W CSE151B_SP26_A00 -g 1 -c 8 -m 32 -s
```

**Extending the 6-hour time limit** (important for Model 3 training): pods are killed after 6 hours by default. Set `K8S_TIMEOUT_SECONDS` *before* running the launch command to extend up to 12 hours:

```bash
export K8S_TIMEOUT_SECONDS=43200   # 12 hours
launch-sp26-cuda128.sh -l gpu-class=medium -W CSE151B_SP26_A00 -g 1 -c 8 -m 32
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
# Should show: A30 (24 GB), CUDA 12.8
```

---

## 4. Clone the repo and set up git credentials

If this is a fresh node (no project files yet):
```bash
cd ~
git clone https://github.com/vishudhshah/cse151b-project.git  # team repo URL "CSE 151B/cse151b-project"
cd "CSE 151B/cse151b-project"
```

If the project folder already exists:
```bash
cd ~/CSE\ 151B/cse151b-project
git pull
```

### Set up git credentials (needed every new pod)

Each DataHub pod is a fresh machine with no SSH keys. Use HTTPS with a stored token:

```bash
git config --global credential.helper store
git remote set-url origin https://github.com/vishudhshah/cse151b-project.git  # team repo URL
```

The first `git push` will prompt for your GitHub username and a personal access token (PAT). Generate one at:
**GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)**
— scope: `repo`. After entering it once, it's stored for the session.

---

## 5. Install dependencies

Run this once per new pod (takes ~3 min):

```bash
cd ~/CSE\ 151B/cse151b-project

# Install uv (fast package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create constraints file to pin torch version
cat > constraints.txt << 'EOF'
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
EOF

# Create venv and install torch first (cu121 — compatible with CUDA 12.8 driver)
uv venv .venv --seed
source .venv/bin/activate
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install everything else
uv pip install sympy numpy transformers vllm tqdm bitsandbytes \
    antlr4-python3-runtime==4.11.1 accelerate peft trl datasets \
    -c constraints.txt

# Set HuggingFace token to avoid rate limiting (get from huggingface.co/settings/tokens)
export HF_TOKEN=hf_your_token_here

# Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.5.1+cu121  True
```

> **Every new session**: you only need `source .venv/bin/activate` and `export PATH="$HOME/.local/bin:$PATH"` — the venv and packages persist on disk between sessions on the same pod. If you get a new pod, re-run the full setup above.

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

> **Progress bar looks stuck on first question?** Normal — the thinking model generates up to 16,384 tokens before answering. The bar only ticks after each complete response. `--max_tokens 2048` makes smoke tests faster; use the default for real runs.

### Full experiment run

Start Model 3 training first (takes ~6 hours) so it runs in the background while you do everything else.

```bash
# ── Step 1: Model 3 training (start first, runs overnight) ────────────────────
nohup python model3_finetune_train.py --epochs 3 --gpu 0 \
    > logs/model3_train.log 2>&1 &
echo "Training PID: $!"

# ── Step 2: Model 1 — all 4 prompt variants (~50 min) ─────────────────────────
nohup python model1_prompt_engineering.py --variant all --gpu 0 \
    > logs/model1.log 2>&1 &
echo "Model 1 PID: $!"

# ── Step 3: Model 2 — temperature sweep (~60 min) ─────────────────────────────
nohup python model2_sampling_voting.py --experiment temp_sweep --gpu 0 \
    > logs/model2_temp.log 2>&1 &

# ── Step 4: Model 2 — majority voting (~60 min per N) ─────────────────────────
nohup python model2_sampling_voting.py --experiment voting_n5 --gpu 0 \
    > logs/model2_vote5.log 2>&1 &

# ── Step 5: Model 3 inference (after training finishes) ───────────────────────
ls checkpoints/model3_qlora/adapter_model.safetensors  # confirm training done
nohup python model3_finetune_infer.py --checkpoint checkpoints/model3_qlora --gpu 0 \
    > logs/model3_infer.log 2>&1 &
```

Monitor any job:
```bash
tail -f logs/model1.log               # live output
ps aux | grep python | grep -v grep   # see all running jobs
kubectl get pod                        # list your pods (from the login node)
kubectl logs <pod-name>                # get output from a background pod
kubectl delete pod <pod-name>          # manually terminate a pod when done
```

> Check GPU availability before launching: [datahub.ucsd.edu/hub/status](https://datahub.ucsd.edu/hub/status) (requires UCSD login).

### Time estimates (A30 GPU)

| Step | Est. time |
|------|-----------|
| Model 1 — all 4 variants | ~50 min |
| Model 2 — temp sweep | ~60 min |
| Model 2 — voting N=5 | ~60 min |
| Model 3 — training (3 epochs) | ~4–6 hours |
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

1. Launch a new pod: `launch-sp26-cuda128.sh -l gpu-class=medium -W CSE151B_SP26_A00 -g 1 -c 8 -m 32`
2. `cd` into the project, `source .venv/bin/activate`
3. Re-run the **exact same command** — it detects already-completed questions and skips them

For training, completed epochs are checkpointed. If killed mid-epoch, re-run from the last saved checkpoint:
```bash
# Resume from the last epoch checkpoint
python model3_finetune_train.py --epochs 3 \
    --output checkpoints/model3_qlora  # trainer resumes automatically
```

---

## 11. Troubleshooting

| Error | Fix |
|-------|-----|
| `launch-sp26-cuda128.sh` hangs on Pending | Use fallback: `launch.sh -v a30 -g 1 -c 8 -m 32` |
| `uv: command not found` | Run `export PATH="$HOME/.local/bin:$PATH"` |
| `source .venv/bin/activate` fails | Venv doesn't exist on this pod — run the full install from Section 5 |
| `torch.cuda.is_available()` is False | CUDA driver mismatch — re-install torch: `uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121` |
| `git@github.com: Permission denied` | Node has no SSH key — switch to HTTPS: `git remote set-url origin https://github.com/vishudhshah/cse151b-project.git  # team repo URL` |
| `No module named 'trl'` | Venv not activated — run `source .venv/bin/activate` |
| `FileNotFoundError: results/...` | `results/` directory missing — run `mkdir -p results logs checkpoints/model3_qlora` |
| Progress bar stuck | Normal for first question — see note in Section 7 |
| HuggingFace download slow | Set `export HF_TOKEN=hf_...` (token from huggingface.co/settings/tokens) |
| Disk quota error during training | Model weights cache is large — `export HF_HOME=/datasets/$USER/hf_cache` |
| Pod status: `OOMKilled` | Ran out of RAM — relaunch with `-m 48` or `-m 64` |
| Pod status: `DeadlineExceeded` | Hit the 6-hour pod limit — set `export K8S_TIMEOUT_SECONDS=43200` before launching (see Section 2) |
