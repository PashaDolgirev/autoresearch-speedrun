# autoresearch-speedrun

Sketching an *agentic research lab*. The big-picture bet: a swarm of
autonomous research agents that, given a tractable training setup and a
clear metric, iterates on language-model-pretraining ideas overnight, and
the human researcher wakes up to a results log and (hopefully) a better
model.

This repo is a small, single-author probe of that bet. It composes two
existing projects:

- **the frame** — Karpathy's [autoresearch](https://github.com/karpathy/autoresearch):
  one file the agent edits, a fixed evaluation, a tight commit-run-keep-or-revert
  loop, and a `program.md` "skill" pointing the agent at it.
- **the substrate** — Keller Jordan's
  [modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt):
  the community benchmark for "how fast can 8× GPUs train a small GPT to
  a target val loss on FineWeb?".

The two are kept in this repo unmodified for reference, under
`autoresearch_original/` and `speedrun_original/`. The actual lab — the
files the agent reads and edits — lives at the repo root.

## What this would look like with real compute

In its full form, the lab would run on the same 8× H100 box the SOTA
speedrun uses, start from the *current world record*
(`speedrun_original/train_gpt.py`, ~1.4 min today), and try to push the
record down further. The agent would propose changes — kernel fusions,
schedule tweaks, architectural ablations, optimizer changes — commit them,
launch real training runs, check val_loss + wall-clock, keep or revert.
Run that loop for a night and you get tens of fully-validated experiments,
all logged, reviewable in the morning.

That's the regime where this idea is interesting. It's not what's running
here today.

## What this is — today

I don't have 8× H100 today. I have access to 8× A100 (40 GB SXM4) for one
night. So this is a deliberately scaled-down instance of the same idea.

Two consequences:

1. **Different baseline.** The current speedrun record (record 80,
   1.4 min) depends on Hopper-only features — FP8 matmul on the LM head,
   Flash Attention 3, TMA-based Triton kernels. None of that runs on
   Ampere. So the baseline here is **modded-nanogpt record 18**
   ([Jan 4, 2025, "Lower logit softcap from 30 to 15"](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2025-01-04_SoftCap)),
   which is the most advanced record that still runs cleanly on A100.
   It uses Muon + Newton–Schulz, FlexAttention, bf16 throughout — no
   FP8, no FA3, no TMA. The full code is at `train.py` at the root.

2. **Different metric.** The full speedrun is "reach val_loss ≤ 3.28 on
   FineWeb val, on 8× H100, as fast as possible". Reaching 3.28 on A100
   with record-18 code takes several minutes per run — too slow for an
   overnight iteration loop. The metric here is shorter and
   compute-honest: **val_loss after a fixed 90 seconds of training**,
   evaluated on the same FineWeb val set. Lower is better.

## The workflow: hypothesize → execute (screen + robust) → record + decide

Each candidate change goes through a **two-stage acceptance funnel**.
The key insight: a single run is noisy. A change can look like a 0.005
improvement just from seed variance. So we compare candidate vs current
best **paired by seed**, and only accept changes that hold up across
many seeds.

```
            ┌─────────────────────────┐
            │  1. Hypothesize         │
            └────────────┬────────────┘
                         │
            ┌────────────▼────────────┐
            │  2a. Edit train.py +    │
            │      git commit         │
            └────────────┬────────────┘
                         │
            ┌────────────▼────────────┐
            │  2b. SCREEN (3 seeds)   │   cheap filter
            │   pass iff cand ≤ best  │
            │   on ALL 3 seeds        │
            └────────────┬────────────┘
                 fail───►│
                         │ pass = PROMISING (not yet accepted)
            ┌────────────▼────────────┐
            │  2c. ROBUST (+7 seeds)  │   expensive validator
            │   pass iff cand ≤ best  │
            │   on ALL 10, ≥1 strict  │
            │   win                   │
            └────────────┬────────────┘
                 fail───►│
                         │ pass = ESTABLISHED
            ┌────────────▼────────────┐
            │  3. Record + KEEP       │
            │     (advance branch)    │
            └─────────────────────────┘

    fail at any stage → DISCARD: record in results.tsv,
                                 git reset --hard HEAD~1
```

Default funnel sizes: **N1 = 3** seeds for screening, **N2 = 7** more
seeds for robustness (10 total). With these numbers, a useless change
("a coin flip vs baseline on each seed"):
- wins all 3 paired screen seeds with probability **~12.5%** (1/8) — fine
  for a coarse filter,
- wins all 10 paired seeds with probability **~0.1%** (1/1024) — matches
  the upstream speedrun's `p<0.01` statistical bar.

So PROMISING ≠ accepted. The robust check is the line.

The end-to-end recipe lives in `program.md`. In short: a fixed-time
`SEED`-controlled training run (`SEED=<n> bash run.sh > run.log 2>&1`),
grep the final `val_loss:` line, compare paired-by-seed to `baseline.tsv`,
keep or revert.

## Design principles — what we thought hard about

This repo is small but it isn't just "autoresearch frame + speedrun
training script glued together". Two of the three workflow steps
required real design effort; this section explains what and why.

### Step 1 (hypothesize) — the dominant bottleneck

In a heavily-optimized regime like the speedrun, **the quality of
hypotheses dominates everything else**. The execution loop processes
candidates at a finite rate (roughly 4–6 fully validated KEEPs and
~50 cheap screen-fails per night on 8× A100); whether those
candidates are well-chosen is the difference between a productive
night and a `results.tsv` full of `discard` rows. The cheap
parameter wins are already baked into record 18 — anything the agent
finds via blind LR / momentum sweeps almost never survives the
10-seed robustness bar.

A strong hypothesis isn't "tune LR by 20%". It requires:

1. **SOTA literacy.** Knowing what's recent, what's been tried, what
   the pretraining community is currently excited about.
2. **Triage.** Quickly assessing which ideas are A100-compatible,
   implementable in `train.py`, and not already in the baseline.
3. **Translation to code.** Mapping a paper's described mechanism
   onto the specific lines of `train.py` to change.
4. **Execution discipline.** Making the smallest defensible change so
   the screen result is actually attributable to the idea.

That list is *intrinsically multi-step*, and in the full agentic-lab
vision it's naturally **multi-agent**: a literature scout that
continuously trawls arXiv and updates an idea pool; an evaluator
that ranks new candidates against the current state of `train.py`
and `results.tsv` (e.g., "optimizer-side ideas have been losing this
week — bias toward attention-side"); an implementer that turns a
paper into a diff; the autoresearch loop itself as the executor. Each
agent has its own context and its own loop.

For this one-night probe we collapsed that pipeline into a single
upfront pass: an arXiv 2025–2026 scan filtered for A100-compatible,
plug-in-able ideas not already in record 18, producing the 25-paper
shortlist at `literature/`. The index file (`literature/README.md`)
deliberately encodes the *editorial* triage layer — for each paper, a
one-line "why this might help here" and a "plug-in surface" mapping
the idea onto a specific function in `train.py`. The autoresearch
agent's in-the-loop reasoning starts from that index, not from a
blank arXiv search.

**This is the part of the repo that is *not* a stitching of two
existing projects.** Everything else descends from autoresearch and
modded-nanogpt; the curated literature pool and the discipline
around using it are the original contribution.

### Step 2 (modify + execute) — robustness over speed

The naive version of step 2 is "edit, run once, check val_loss, keep
if better". For pretraining at this scale this fails catastrophically:
a single run's val_loss has roughly 0.01–0.02 standard deviation just
from seed noise, and real per-step improvements often live well under
that bar. Accept a single-run win and you accumulate false positives
forever — the recorded "best" drifts up while the actual training
quality drifts down.

The upstream speedrun rules already encode this: a new record must
be established at `p<0.01` significance vs. the prior best across
multiple runs. We adapt that into a cheap **screen + robust funnel**:

- **Screen (N₁ = 3 paired-seed runs)**: a useless change wins all 3
  by noise alone ~12.5% of the time. Cheap filter that catches
  obvious losers in ~6 minutes per candidate.
- **Robust (N₂ = 7 additional paired-seed runs, 10 total)**: a useless
  change wins all 10 ~0.1% of the time — matching the upstream
  `p<0.01` bar.

The promotion gate is explicit: a candidate that passes the screen
is labelled **PROMISING**, but `baseline.tsv` is *not* updated until
robust passes. That separation is the single most important detail
of step 2 — without it, the screen would silently leak false
positives into the record of "current best", and the agent's
subsequent decisions (rescue-tweak budget, what to compare against,
what to revert to) would all be poisoned.

The framework contract in `train.py` (the `TIME_BUDGET_S` + `SEED`
block, marked `DO NOT MODIFY`) exists so that paired-seed
comparisons are actually comparable. It's deliberately small —
three blocks of code — and out of the agent's edit surface, so the
agent doesn't accidentally make its own runs incomparable.

### Step 3 (record + decide) — institutional memory

Mostly bookkeeping, but with one subtlety: step 3's outputs are
step 1's *inputs* on the next iteration.

- `baseline.tsv` is overwritten on every KEEP — it's the live
  per-seed reference the next candidate is compared against.
- `results.tsv` is append-only — including the rescued-and-still-
  failed attempts that wouldn't appear in `git log` because they
  got reset away. This makes the experimental history queryable
  for the agent's own use.

The capped-rescue rule in step 1 ("if a promising idea fails screen
by < 0.01 at one seed, try ≤ 2 hyperparameter adjustments before
giving up") only works because step 3 makes the history queryable.
Without `results.tsv`, the agent would re-attempt the same dead-end
rescues over and over.

### What's novel here

To summarize what the repo contributes beyond the two upstream
projects under `autoresearch_original/` and `speedrun_original/`:

1. **A two-stage paired-seed acceptance funnel** with explicit
   PROMISING vs. ESTABLISHED separation, calibrated so the robust
   gate matches the upstream `p<0.01` statistical bar.
2. **A pre-curated literature shortlist** (`literature/`) acting as
   the single-pass substitute for the multi-agent hypothesis
   pipeline, filtered for A100-compat and "not already in the
   baseline".
3. **An A100-honest baseline.** Record 18 instead of the current
   record, so the autoresearch loop iterates in the regime where
   the compute we actually have lets an experiment finish in 2–3
   minutes.
4. **An explicit framework contract in `train.py`** that makes
   paired-seed comparisons reproducible without removing the
   agent's freedom to edit the rest of the file.

Everything else (the train script itself, the autoresearch
run-edit-keep loop, the modded-nanogpt training recipe) is
borrowed directly from upstream.

## Files at the root

### Agent-visible

| File | Editable? | Purpose |
|---|---|---|
| `train.py` | **yes**, except a tiny `DO NOT MODIFY` framework-contract block (SEED + TIME_BUDGET_S + the time-up early stop) | record-18 modded-nanogpt; single agent-editable file |
| `prepare.py` | no | one-time FineWeb10B GPT-2-token download into `./data/fineweb10B/` |
| `run.sh` | no | wraps `torchrun ... train.py`, passes `SEED` and `TIME_BUDGET_S` |
| `run_with_retry.sh` | no | wraps `bash run.sh` with a 240 s watchdog + retries; agent should always call this, not bare `run.sh` |
| `requirements.txt` | no | deps |
| `program.md` | human-edited | the agent's skill / instructions |
| `literature/` | read-only | curated 25-paper 2025–2026 arXiv shortlist for hypothesis formation (PDFs + `literature/README.md` index, grouped by plug-in surface) |
| `baseline.tsv` | overwritten on every KEEP (gitignored) | per-seed val_loss of the current best (one row per seed, 10 rows) |
| `results.tsv` | append-only (gitignored) | one row per stage-decision per candidate |
| `.gitignore` | — | covers `baseline.tsv`, `results.tsv`, `logs/`, `data/`, `run.log` |

### Reference (do not modify)

| Directory | What |
|---|---|
| `autoresearch_original/` | Karpathy's autoresearch — the frame |
| `speedrun_original/` | Keller Jordan's modded-nanogpt — substrate, current SOTA train script, full record history in `speedrun_original/README.md` |

## Quick start (on 8× A100)

```bash
# 0. One-time: make this a clean git repo on the box.
#    The agent uses `git commit` / `git reset --hard HEAD~1` to advance and
#    revert candidates, so we need at least one commit on the parent repo.
#    The nested .git folders in autoresearch_original/ and speedrun_original/
#    (left over from the original `git clone`s) confuse `git add` — git would
#    treat them as submodule pointers and silently drop the reference file
#    contents. Strip them before the first commit.
rm -rf autoresearch_original/.git speedrun_original/.git
[ -d .git ] || git init
git add -A
git commit -m "baseline (record 18 verbatim + autoresearch frame)"

# 1. Install deps
pip install -r requirements.txt

# 2. Download data (~900M tokens, a few minutes)
python prepare.py

# 3. Calibrate: run the unmodified baseline at seeds 0..9.
#    Use `run_with_retry.sh`, NOT bare `run.sh` — there's a residual ~20%
#    NCCL hang rate on this A100 + torch 2.7.1 + NCCL 2.26.2 combo, and the
#    retry wrapper kills+retries hung runs at a 240 s wall-clock cap.
#    First run takes ~3-5 min for torch.compile; subsequent runs ~95 sec
#    each (plus a few extra runs from hang retries). Total ~25-30 min.
for s in 0 1 2 3 4 5 6 7 8 9; do
    SEED=$s bash run_with_retry.sh > run.log 2>&1
    echo "seed=$s: $(grep 'val_loss:' run.log | tail -1)" | tee -a calibration.txt
done

# 4. Write the 10 (commit, seed, val_loss_at_90s) rows into baseline.tsv
#    (schema in program.md). Append a 'baseline' row to results.tsv.

# 5. Point your agent at program.md and let it go.
#    I run Claude Code with `--dangerously-skip-permissions` for this so
#    the agent never blocks on a permission prompt for `bash`, `git`,
#    `Edit`, etc. — overnight autonomous operation requires no human in
#    the loop. The rented A100 box is ephemeral and only has the FineWeb
#    data + this repo on it, so the blast radius of "skip permissions" is
#    contained to the experiment itself.
```

## Checking in on the run

Everything is plain text on disk — you can `ssh` in anytime and see the
state without interrupting the agent. Nothing is in-memory-only. The
agent itself runs with `--dangerously-skip-permissions`, so it makes
progress without waiting on you; you check in when you want to, not
because it's blocked.

### What's persisted, and when it changes

| File | Updates | What you see |
|---|---|---|
| `results.tsv` | append-only, one row per stage-decision per candidate | every candidate's outcome: commit, stage (`screen`/`robust`), seeds_passed (e.g. `3/3`, `9/10`), mean/max val_loss, status (`promising` / `keep` / `discard` / `crash`), description |
| `baseline.tsv` | overwritten on every KEEP | per-seed val_loss of the *current best* (10 rows). Numbers drift down over time as ESTABLISHED candidates accumulate. |
| `git log` (current branch) | one commit per ESTABLISHED candidate. Failed candidates are `git reset --hard HEAD~1`'d, so they don't appear here. | the chain of accepted improvements only, each with the agent's commit message |
| `run.log` | overwritten *per run* | the currently running experiment's live stdout (val_loss / train_time / step lines, peak memory at the end) |
| `logs/<uuid>.txt` | one file per run, ever (uuid-named, never overwritten) | every run's full log, including the snapshot of `train.py` that was used for that run |

### What to look at, fast

```bash
cd autoresearch-speedrun

# Current best — per-seed val_loss of the accepted baseline.
cat baseline.tsv

# Every candidate's outcome. Append-only; tail for recent.
tail -30 results.tsv

# Accepted-improvement chain: each non-revert commit is one win.
git log --oneline -20

# Counts by outcome.
awk -F'\t' 'NR>1 {print $6}' results.tsv | sort | uniq -c
```

### Watching a live run

```bash
# Whatever's training right now (run.log is overwritten each run).
tail -f run.log

# Or the per-run persistent log of the most recent run:
tail -f "logs/$(ls -t logs/ | head -1)"
```

### Is anything happening right now?

```bash
ps aux | grep -E "torchrun|train\.py" | grep -v grep
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```

`results.tsv` and `git log` together tell the whole story over coffee — the
ledger of attempts and the ledger of wins, respectively.

## Results — first overnight run (2026-05-16/17, 8× A100 40 GB, ~6.7 h)

Headline numbers, paired across the 10 calibration seeds:

| | Baseline (record 18 + A100 shims, commit `6385af7`) | Final (commit `ef53b87`) | Δ |
|---|---|---|---|
| Mean val_loss @ 90 s | **3.9249** | **3.8093** | **−0.116** |
| Max across 10 seeds | 3.9346 | 3.8202 | −0.114 |
| Min across 10 seeds | 3.9177 | 3.7986 | −0.119 |
| Spread (max − min) | 0.0169 | 0.0216 | (similar) |

That's ~6.8× the per-run seed-noise spread — real signal, not chance. Six
candidates passed the 10-seed paired-acceptance bar:

| # | Commit | Change | Source | Δ |
|---|---|---|---|---|
| 1 | `5382153` | EMA at eval, decay=0.9 | literature ([arXiv 2502.06761](https://arxiv.org/abs/2502.06761)) | **−0.0730** |
| 2 | `881444e` | `num_iterations` 1390 → 400 | own reasoning (schedule realignment) | −0.0072 |
| 3 | `d1d3c7e` | `cooldown_frac` 0.4 → 0.85 | own reasoning | −0.0265 |
| 4 | `6f7d132` | EMA decay 0.9 → 0.95 (re-tune) | literature follow-up | −0.0037 |
| 5 | `6ab97db` | linear → cosine LR cooldown | standard ML | −0.0017 |
| 6 | `ef53b87` | `embed_lr` 0.6 → 0.5 (rescue) | parameter sweep | −0.0036 |

About **30 candidates were considered** in total; 6 accepted (~20 %
acceptance rate), the rest discarded at screen or robust. Roughly $107 of
compute at $15.92/hr.

### What worked

- **The single literature pick that landed paid for the night.** EMA at
  eval alone is 63 % of the total gain. The 90-s budget ends mid-noisy-
  training; averaging the recent weights before the final eval de-noises
  the prediction more than any single architectural change did.
- **Schedule realignment to the *actual* training duration was the
  biggest agent-original insight** (#2, #3, #5 — combined Δ ≈ −0.035).
  Record 18's `num_iterations=1390`, `cooldown_frac=0.4`, and sliding-
  window warmup are tuned for the full speedrun budget; at our 90-s A100
  budget we only reach step ~265, so cooldown never triggers and the
  sliding window stays tiny. Compressing the schedule made each
  mechanism actually fire within the budget.

### What didn't work

The architectural literature picks tried at this short-horizon regime
**all failed** (Cautious Weight Decay, Peri-LN, V-norm / HybridNorm,
Muon-VS, MTP-lite). Several lost by wide margins (Peri-LN +0.089, MTP
+0.052). This is itself a useful finding: **record 18 is so heavily
tuned that net-new architectural changes destabilize more than they help
in ~265 training steps**. The architectural-leverage regime starts at
longer horizons.

### The funnel did its job

The strict 10-seed paired bar caught at least one near-miss that a
naive single-run accept would have promoted:

- `3808a0b` (embed_lr 0.6 → 0.45) screened 3/3 promising but **failed
  robust 9/10**, losing seed 7 by 0.0007. Correctly discarded.
- The agent then ran the documented rescue (embed_lr 0.6 → 0.5, smaller
  step), which passed 10/10 and became KEEP #6.

Without the PROMISING → ESTABLISHED separation, `3808a0b` would have
overwritten `baseline.tsv` with a baseline that was unfairly low on seed
7, and every subsequent candidate would have been measured against a
poisoned reference. This is the design contribution doing visible work.

Full per-candidate log: `box_results/autoresearch-speedrun/results.tsv`
(gitignored) — pulled back from the box after the run.

## Status

- [x] Frame staged: record 18 at root, framework contract wired into `train.py`.
- [x] Two-stage funnel (`screen` 3 seeds → `robust` +7 seeds) documented in `program.md`.
- [x] Baseline calibrated on 8× A100 (mean val_loss 3.9249 at 90 s).
- [x] First overnight agent run: 6 ESTABLISHED wins, val_loss 3.9249 → 3.8093 (−0.116) in ~6.7 hours.

## License

Inherits the upstream licenses (MIT for both autoresearch and modded-nanogpt).
