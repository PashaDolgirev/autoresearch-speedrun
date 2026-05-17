# autoresearch — A100 speedrun edition

This is an experiment to have an LLM do its own research, applied to a
**scaled-down version of the [modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt)
that runs on 8× A100 GPUs**.

Background and rationale are in `README.md`. The short version: we don't
have 8× H100s, so we can't run the current record code (it depends on FP8
matmul, Flash Attention 3, and TMA-based Triton kernels — all Hopper-only).
The baseline here is **modded-nanogpt record 18** (Jan 4, 2025, "Lower
logit softcap from 30 to 15"), which is the most advanced record that runs
cleanly on Ampere.

**The metric.** Every run is a fixed 90-second wall-clock training session
followed by a final `val_loss` eval on the FineWeb val set. Each run is
controlled by a `SEED` env var. The harness compares candidate vs. baseline
**paired by seed**, and only accepts changes that hold up across many seeds.

The agent edits a single file (`train.py`) and runs experiments in a loop:
hypothesize → modify → execute (screen → robust) → record + decide.

## Workflow at a glance

```
            ┌─────────────────────────┐
            │  1. Hypothesize         │  ← pick one idea
            └────────────┬────────────┘
                         │
            ┌────────────▼────────────┐
            │  2a. Edit train.py +    │  ← agent edits
            │      git commit         │
            └────────────┬────────────┘
                         │
            ┌────────────▼────────────┐
            │  2b. SCREEN (3 seeds)   │  ← seeds 0,1,2
            │   pass iff cand ≤ best  │
            │   on ALL 3 seeds        │
            └────────────┬────────────┘
                 fail───►│
                         │ pass = PROMISING
            ┌────────────▼────────────┐
            │  2c. ROBUST (+7 seeds)  │  ← seeds 3..9
            │   pass iff cand ≤ best  │
            │   on ALL 7, ≥1 strict   │
            │   win                   │
            └────────────┬────────────┘
                 fail───►│
                         │ pass = ESTABLISHED
            ┌────────────▼────────────┐
            │  3. Record + KEEP       │  ← update baseline.tsv,
            │     (advance branch)    │     append results.tsv
            └─────────────────────────┘

    fail at any stage → DISCARD: record in results.tsv,
                                 git reset --hard HEAD~1
```

**Numbers** (configurable, current defaults: `N1 = 3, N2 = 7`):
- A useless change wins 3 paired seeds by pure noise ~12.5% of the time
  (1/8) — fine for a coarse screen.
- A useless change wins 10 paired seeds by pure noise ~0.1% of the time
  (1/1024) — matches the upstream speedrun's "p<0.01" statistical bar.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may16`).
   The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from `main`.
3. **Read the in-scope files**. The repo is small:
   - `README.md` — repository context.
   - `program.md` — this file.
   - `prepare.py` — one-time FineWeb10B download. **Do not modify.**
   - `run.sh` — wraps `torchrun --standalone --nproc_per_node=8 train.py`,
     passes through `SEED` and `TIME_BUDGET_S`. **Do not modify.**
   - `run_with_retry.sh` — the wrapper you should actually call. Wraps
     `bash run.sh` with a 240 s wall-clock watchdog + up to 2 retries to
     paper over a residual ~20% NCCL hang rate on this A100 + torch combo.
     Without it, ~1 in 5 seeds hangs silently and you waste a seed.
     **Do not modify.**
   - `train.py` — the file you modify. Model, optimizer, schedule, training
     loop. Verbatim record-18 modded-nanogpt **except** for a short
     framework-contract block (the `TIME_BUDGET_S` early stop, the `SEED`
     wiring) clearly marked `DO NOT MODIFY`. Everything else is fair game.
   - `speedrun_original/` — pristine upstream modded-nanogpt repo, including
     the current world-record code and the full record history in
     `speedrun_original/README.md`. **Read-only context**. Many later-record
     ideas depend on Hopper-only features (FP8, FA3, TMA) and won't run on
     A100 — when in doubt, lean on ideas from records 12–18.
   - `autoresearch_original/` — Karpathy's autoresearch repo (the frame).
     Read-only.
4. **Verify data exists**: Check that `./data/fineweb10B/` contains
   `fineweb_val_000000.bin` and at least 9 `fineweb_train_*.bin` files. If
   not, tell the human to run `python prepare.py`.
5. **Initialize `results.tsv`**: Create `results.tsv` with the header row
   (see schema below).
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, **first run calibration** (next section), then
kick off the experimentation loop.

## Calibration (once per machine, before any agent experiments)

The point: populate `baseline.tsv` — the current-best (initially: vanilla
record-18) per-seed val_loss at 90 sec on this machine.

```
for SEED in 0 1 2 3 4 5 6 7 8 9; do
    SEED=$SEED bash run_with_retry.sh > run.log 2>&1
    # extract the final val_loss line (the one printed at the time-budget stop)
    grep "val_loss:" run.log | tail -1
done
```

After all 10 runs, write `baseline.tsv` (tab-separated, gitignored):

```
commit	seed	val_loss_at_90s
a1b2c3d	0	3.8721
a1b2c3d	1	3.8650
a1b2c3d	2	3.8810
a1b2c3d	3	3.8702
a1b2c3d	4	3.8591
a1b2c3d	5	3.8784
a1b2c3d	6	3.8633
a1b2c3d	7	3.8770
a1b2c3d	8	3.8688
a1b2c3d	9	3.8745
```

(Numbers above are illustrative — real values come from your runs.) All 10
rows must share the **same `commit`** (the head of the setup branch,
unmodified record-18).

Also append the baseline summary to `results.tsv`:

```
commit	stage	seeds_passed	mean_val_loss	max_val_loss	status	description
a1b2c3d	baseline	10/10	3.8709	3.8810	baseline	record 18 verbatim — calibration
```

The first compile takes ~3-5 min; subsequent runs reuse the compile cache.
Total wall-clock for the 10 calibration runs: ~25-30 min.

## Experimentation

Each experiment is launched via `SEED=<n> bash run_with_retry.sh > run.log 2>&1`.
That wrapper invokes `bash run.sh` under a 240 s watchdog and retries up to 2
times if the run hangs (NCCL deadlock; happens ~20% of the time on this
hardware, see `run_with_retry.sh` header for details). The underlying
training script writes a per-run log to `logs/<uuid>.txt` and prints key
lines to console. The final `val_loss:` line in the log is the value at the
90-sec budget stop — that's the number you score.

**Always use `run_with_retry.sh`, never bare `bash run.sh`** — a bare call
will sit forever if NCCL hangs, silently burning your time budget for the
night.

**What you CAN do:**
- Modify `train.py` — model architecture, optimizer, hyperparameters,
  schedule, batch size, sequence length, attention pattern, validation
  cadence, kernel choices, etc.

**What you CANNOT do:**
- Modify the framework contract block in `train.py` (it's marked
  `DO NOT MODIFY`). It controls `SEED`, `TIME_BUDGET_S`, and the time-up
  early stop. Without it, runs aren't comparable.
- Modify `prepare.py`, `run.sh`, or anything in `data/`.
- Change the data pipeline (the stream of tokens is fixed).
- Change the validation harness (the existing `val_tokens = 10_485_760`
  and the eval loop). You may change *eval cadence* (`val_loss_every`)
  freely; only the final time-up eval is scored.
- Use any extra `torch._inductor.config` or `torch.compile` flags beyond
  what the baseline already has. (Banned by upstream speedrun rule #3.)
- Install new packages.
- Use Hopper-only features (FP8 matmul, FA3, TMA Triton kernels). They
  won't run on A100.

**VRAM** is a hard constraint — A100 is 40 GB. Baseline uses ~12–18 GB;
doubling memory could OOM. If a run OOMs, log `crash` and revert.

**Simplicity criterion**: All else being equal, simpler is better. A small
gain that adds ugly complexity is not worth it. Removing something and
getting equal or better performance is a great outcome — a simplification
win. Weigh complexity vs. magnitude.

**Idea sources**:
- The record table in `speedrun_original/README.md` lists every technique
  that moved the upstream record, with PR links. Records 12–18 are the
  directly relevant pre-Hopper progression. Some later-record ideas (some
  schedule / momentum tweaks, small architectural ablations) also work on
  A100 if they don't depend on FP8/FA3/TMA.
- Karpathy's `autoresearch_original/train.py` has general tricks
  (resid/x0 lambdas, value-embedding gates, cautious weight decay
  schedule) worth borrowing.

## The experiment loop

LOOP FOREVER:

### 1. Hypothesize

The speedrun is a heavily-optimized regime. The record-18 baseline you start
from already incorporates ~7 generations of community-curated improvements
(Muon, Newton-Schulz orthogonalization, FlexAttention with sliding-window
schedule, value embeddings + U-net structure, ReLU², QK-norm, zero-init
projections, half-truncate RoPE, ...). The cheap wins are already in the
baseline. That has consequences for how you should pick hypotheses:

**1) Parameter sweeps and small tweaks are *weak* research in this regime.**
LR ±20%, momentum 0.95→0.97, slightly different cooldown frac, etc. are
fine *as a closing move on a stronger idea*, but they are a weak primary
move. They almost never survive the 10-seed robustness bar on their own
because the baseline is already near a local optimum on those knobs. **The
goal is a strong research step, not a strong fitting step.**

**2) Lean on architectural / algorithmic ideas — seriously, and from the
literature.** What has historically moved the speedrun is *structural*
change: new optimizer variants, new attention shapes, new precision
regimes, new parameterizations, novel loss formulations. A "strong
hypothesis" has a defensible mechanistic reason it should improve either
compute-per-step or per-step learning efficiency, ideally backed by a
published result.

A curated shortlist of recent (2025–2026) arXiv papers relevant to this
exact setup lives in **`literature/`**:
- Open **`literature/README.md` first**. It's a one-screen-per-paper index:
  title, arXiv link, 1-line key claim, 1-line "why it might help here",
  1-line "where in `train.py` it would plug in", 1-line A100-compat notes.
  Entries are grouped by plug-in surface (Optimizer / Attention /
  Architecture / Schedule / Loss / Numerical).
- **Skim** the index. **Do not** open PDFs while skimming.
- When an idea sounds genuinely promising, *only then* open the corresponding
  PDF (`literature/<arxiv-id>.pdf`) for the specific technical details you
  need to implement it.
- **Do not write a summary of any paper you read.** The index is the
  summary. Burning context on re-summarization is exactly what the index
  exists to prevent. If the index entry is wrong about something
  important you discovered, just note the correction inside your
  `results.tsv` description — don't author a new summary file.

**3) If a strong idea fails the screen, try a *small* parameter-tweak
rescue before fully abandoning it.** A new architectural mechanism often
needs a small init / LR / momentum shift to be apples-to-apples with the
baseline; the baseline's hyperparameters were tuned for the baseline's
architecture. So if the candidate looked plausible on paper but failed
the 3-seed screen, ask: was it close (lost a single seed by < ~0.01)?
If yes, consider one or two principled hyperparameter adjustments
*around the new mechanism* (lower init scale, slightly higher LR for the
new params, etc.) and re-screen. Use `results.tsv` and `git log` as
institutional memory: a previous failed attempt at the same family of
ideas tells you which knobs have been tried. **Cap this at 2 rescue
attempts per architectural idea** — if it doesn't survive after that,
move on. The screen is supposed to be cheap; rescue attempts are still
3 runs each.

**4) Make the smallest change that tests the idea.** Whatever the source
of the hypothesis, edit `train.py` minimally — one substantive change at a
time. Bundling 3 changes makes the screen result un-attributable. If two
changes genuinely belong together (e.g. a new optimizer + its specific
init scheme), commit them as one experiment; otherwise split them into
sequential experiments.

### 2. Modify and execute

**2a. Edit + commit.** Edit `train.py` directly.
`git commit -am "<short description>"`.

**2b. SCREEN — run at seeds 0, 1, 2.** For each seed:
```
SEED=<seed> bash run_with_retry.sh > run.log 2>&1
grep "val_loss:" run.log | tail -1     # the final eval line at ~90 sec
grep "peak memory" run.log | tail -1
```
Extract `val_loss` (last `val_loss:` line) and `peak_mem_mib` (the
"peak memory consumption: X MiB" line). If `grep "val_loss:"` is empty,
the run crashed — `tail -n 80 run.log`, try a quick fix or mark `crash`.

After all 3 seeds:
- **Pass screen** iff `candidate.val_loss[s] ≤ baseline[s]` from
  `baseline.tsv` for **every** s ∈ {0, 1, 2}.
- **Fail screen** → append a `discard` row to `results.tsv`, then
  `git reset --hard HEAD~1`. Go back to step 1.

If passed: status is **PROMISING**. Append a `promising` row to
`results.tsv` (for visibility) and continue.

**2c. ROBUST — run at seeds 3, 4, 5, 6, 7, 8, 9** (7 more seeds, total 10):
Same procedure. After all 7 seeds:
- **Pass robust** iff `candidate.val_loss[s] ≤ baseline[s]` for every
  s ∈ {0..9}, **and** there is at least one strict win
  (`candidate.val_loss[s] < baseline[s]` for some s). Pure 10-way ties
  are `discard` — no improvement, not worth the churn.
- **Fail robust** → append `discard` row, `git reset --hard HEAD~1`. Note
  the candidate had passed screen but failed robust — log it. Go back
  to step 1.

### 3. Record and decide

If the candidate passed both stages: **KEEP**.
- Overwrite `baseline.tsv` with the candidate's 10 seed val_losses
  (commit field = the new commit hash).
- Append a final `keep` row to `results.tsv`.
- Branch advances; go back to step 1.

If the candidate failed at any stage: **DISCARD**.
- Append a `discard` (or `crash`) row to `results.tsv`.
- `git reset --hard HEAD~1`. Branch returns to previous best.
- Go back to step 1.

## File schemas

### `baseline.tsv` (gitignored, overwritten on every KEEP)

Per-seed val_loss of the current best at 90 sec.

```
commit	seed	val_loss_at_90s
```

- `commit`: short (7-char) hash of the current best — same across all rows.
- `seed`: integer, 0 through 9.
- `val_loss_at_90s`: 4 decimals.

Exactly 10 data rows (one per seed). When KEEP happens, this file is
rewritten with the new commit's seed val_losses.

### `results.tsv` (gitignored, append-only)

One row per stage-decision per candidate.

```
commit	stage	seeds_passed	mean_val_loss	max_val_loss	status	description
```

- `commit`: short hash of the candidate.
- `stage`: `baseline` | `screen` | `robust`.
- `seeds_passed`: `M/N` (e.g. `3/3`, `2/3`, `10/10`, `9/10`).
- `mean_val_loss`: mean over the seeds run in this stage, 4 decimals.
  `0.0000` for `crash`.
- `max_val_loss`: max over the seeds run in this stage, 4 decimals.
  `0.0000` for `crash`.
- `status`: `baseline` | `promising` | `keep` | `discard` | `crash`.
- `description`: short text. **No tabs**.

Status transitions per candidate:
- Crashed before any clean run: one `crash` row, stage `screen`.
- Failed screen: one `discard` row, stage `screen`, with `seeds_passed`
  like `2/3` or `0/3`.
- Failed robust: a `promising` row (stage `screen`, `3/3`) followed by a
  `discard` row (stage `robust`, e.g. `9/10`).
- Accepted: a `promising` row, then a `keep` row
  (stage `robust`, `10/10`).

Example:

```
commit	stage	seeds_passed	mean_val_loss	max_val_loss	status	description
a1b2c3d	baseline	10/10	3.8709	3.8810	baseline	record 18 verbatim — calibration
b2c3d4e	screen	2/3	3.8740	3.8830	discard	muon ns_steps 5 -> 4 (lost seed 1)
c3d4e5f	screen	0/3	0.0000	0.0000	crash	doubled batch_size -> OOM
d4e5f6g	screen	3/3	3.8612	3.8680	promising	muon momentum warmup 300 -> 200 steps
d4e5f6g	robust	9/10	3.8645	3.8720	discard	failed at seed 6 (3.8810 > 3.8745)
e5f6g7h	screen	3/3	3.8588	3.8650	promising	cooldown_frac 0.4 -> 0.45
e5f6g7h	robust	10/10	3.8602	3.8721	keep	cooldown_frac 0.4 -> 0.45
```

## Timing and crash handling

**Per-run wall clock**: ~3-5 min on first compile, ~95 sec on a clean run
thereafter (90 sec of training + ~5 sec val + eval overhead). With the
240 s watchdog retry in `run_with_retry.sh`, the worst-case per-seed cost
is ~3 × ~95 sec ≈ 5 min if every attempt hangs (probability ~0.8%); typical
seeds take 1 attempt (~95 sec) or sometimes 2 (~3 min). The compile cache
survives across runs in the same Python environment.

**Per-candidate wall clock**:
- DISCARD (screen fail): 3 runs ≈ 6-9 min.
- DISCARD (robust fail): 10 runs ≈ 20-30 min.
- KEEP: 10 runs ≈ 20-30 min.

**Crashes**: If a run crashes (OOM, NCCL error, typo, missing import), use
your judgment. If it's a dumb fix (typo, import), fix and re-run. If the
idea itself is fundamentally broken, mark `crash` and move on.

**Hard timeout**: If a single run exceeds **15 minutes** of wall-clock
total, kill it and treat it as `crash`. Also kill any run whose `step:N/M`
log has not advanced for **2 minutes**.

**NEVER STOP**: Once the experiment loop has begun (after setup and
calibration), do NOT pause to ask the human if you should continue. Do
NOT ask "should I keep going?" or "is this a good stopping point?". The
human might be asleep, or gone from the computer and expects you to
continue working *indefinitely* until manually stopped. You are
autonomous. If you run out of ideas, think harder — read PR links in
`speedrun_original/README.md`, re-read the in-scope files for new angles,
try combining previous near-misses, try more radical architectural
changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep.
If a typical screen-fail takes ~7 min and a full KEEP takes ~25 min, then
over a 10-hour night you can screen ~50–60 candidates with ~3–5 fully
established improvements. The user wakes up to a `results.tsv` log of
experiments and a `baseline.tsv` that hopefully has lower numbers than
they started the night with.
