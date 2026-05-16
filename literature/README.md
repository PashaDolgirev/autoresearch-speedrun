# Literature shortlist for autoresearch-speedrun (A100 edition)

This is a curated list of 25 arXiv papers (2025–2026) selected as a
high-leverage idea-pool for the autonomous-research agent iterating on
modded-nanogpt **record 18** on 8× A100. The agent should consult this
index *first* when forming a hypothesis (see `../program.md` step 1),
and only open a paper's PDF when its index entry looks promising.

**Filters used.** Papers from 2025–2026 (one 2024-10 seed reference for
nGPT, retained for its outsized claimed effect), applicable to dense
decoder-only LM pretraining, A100-compatible (no FP8, no FA3, no TMA),
implementable within a few hundred lines of `train.py`, and **not
already implemented in the record-18 baseline** (so the original Muon
paper, original value-residual paper, original ReLU², original QK-norm,
original FlexAttention-sliding-window paper, etc. are deliberately
excluded — those are *already in the code*).

**How to read an entry.** "Key claim" is the paper's own headline.
"Why it might help here" is editorial — the curator's read on how this
specific paper plugs into record 18 and could plausibly move 90-s
val_loss. "Plug-in surface" tells you where in `train.py` the change
would land.

## Optimizer & training dynamics

### Newton-Muon Optimizer
- **arXiv:** [2604.01472](https://arxiv.org/abs/2604.01472) — 2026-04
- **Key claim:** A Newton-style replacement for the Newton-Schulz iteration in Muon: replaces fixed (a,b,c) cubic coefficients with an adaptive Newton-like update on the matrix sign, claimed to converge faster than NS5 with similar per-step cost.
- **Why it might help here:** Record 18 spends real wall-clock per step on 5 Newton-Schulz iterations inside Muon. A drop-in faster orthogonalization translates directly into more training steps in 90 s.
- **Plug-in surface:** `zeropower_via_newtonschulz5` in `train.py`.
- **A100 notes:** Clean — pure bf16 matrix-matrix ops, no Hopper-specific intrinsics.
- **File:** `2604.01472.pdf`

### Variance-Adaptive Muon (Muon-NSR / Muon-VS)
- **arXiv:** [2601.14603](https://arxiv.org/abs/2601.14603) — 2026-01
- **Key claim:** Two Muon variants that apply variance-adaptive normalization to momentum *before* orthogonalization — Muon-NSR uses noise-to-signal ratio modulation, Muon-VS uses variance-based scaling. Both report faster pretraining than vanilla Muon without extra hyperparameters.
- **Why it might help here:** Slot-in change to the existing Muon `step()` that tackles a known failure mode (high-variance momentum corrupting the orthogonalization). Zero-extra-hyperparameter version (VS) is ideal for the strict 10-seed acceptance bar.
- **Plug-in surface:** `Muon.step` momentum-buffer handling, just before the `zeropower_via_newtonschulz5` call.
- **A100 notes:** Clean — operates on existing bf16 buffers.
- **File:** `2601.14603.pdf`

### Effective Quantization of Muon Optimizer States (8-bit Muon)
- **arXiv:** [2509.23106](https://arxiv.org/abs/2509.23106) — 2025-09
- **Key claim:** Blockwise int8 quantization of Muon's momentum buffers matches full-precision Muon in val loss on Chinchilla-optimal GPT pretraining up to 2.7B, while substantially reducing optimizer-state memory.
- **Why it might help here:** A100 40 GB is a hard cap. Freed VRAM lets the agent try larger micro-batches / longer sequences / wider attention windows — all of which can help 90-s val loss even when the optimizer itself is identical.
- **Plug-in surface:** `Muon.__init__` (state allocation) + `step()` (dequantize/requantize).
- **A100 notes:** Clean — int8 quant is broadly supported on Ampere.
- **File:** `2509.23106.pdf`

### MuonBP: Block-Periodic Orthogonalization
- **arXiv:** [2510.16981](https://arxiv.org/abs/2510.16981) — 2025-10
- **Key claim:** Apply orthogonalization independently to matrix shards on each device most of the time, with periodic full orthogonalization to maintain stability — faster than vanilla Muon's full-matrix NS at scale.
- **Why it might help here:** Record 18's Muon does a custom dist `all_gather` to share orthogonalized chunks. Block-periodic should reduce that comm at minimal accuracy cost — direct overlap with the existing distributed-Muon plumbing.
- **Plug-in surface:** `Muon.step` distributed loop (`for base_i in range(...)[::self.world_size]`).
- **A100 notes:** Clean. Same NCCL primitives.
- **File:** `2510.16981.pdf`

### Understanding and Improving Shampoo and SOAP via KL Minimization
- **arXiv:** [2509.03378](https://arxiv.org/abs/2509.03378) — 2025-09
- **Key claim:** Reframes Shampoo and SOAP as KL minimization over structured second-moment estimators. Introduces a corrected variant evaluated on NanoGPT (124M), NanoRWKV (162M), Llama-style (134M), and NanoMoE (227M).
- **Why it might help here:** A serious alternative-optimizer test — SOAP-style methods have been competitive with Muon on small-scale pretraining. The 124M NanoGPT setup in the paper is directly comparable to record 18.
- **Plug-in surface:** Replace `Muon` for hidden-matrix params (or hybrid: keep Muon, swap SOAP into embedding/head bank).
- **A100 notes:** Clean — eigen-decomposition can be done in fp32 on Ampere.
- **File:** `2509.03378.pdf`

### Benchmarking Optimizers for LLM Pretraining
- **arXiv:** [2509.01440](https://arxiv.org/abs/2509.01440) — 2025-09
- **Key claim:** Head-to-head benchmark of AdamW vs. ADOPT, Prodigy, AdEMAMix, SF-AdamW, D-Muon, SOAP, etc. on LLM pretraining. ADOPT, AdEMAMix, D-Muon, and SF-AdamW all outperform AdamW.
- **Why it might help here:** Record 18 uses Adam for embeddings/head/scalars and Muon for matrix params. AdEMAMix or SF-AdamW are drop-in for the Adam side of the optimizer table — much smaller risk than replacing Muon.
- **Plug-in surface:** `optimizer1 = torch.optim.Adam(...)` block. (Some variants may need a few lines of state init.)
- **A100 notes:** Clean — all PyTorch-implementable optimizers.
- **File:** `2509.01440.pdf`

### Minimalist Optimizer Design / SCALE
- **arXiv:** [2506.16659](https://arxiv.org/abs/2506.16659) — 2025-06
- **Key claim:** Two minimal tricks: (a) column-wise gradient normalization, which makes momentum-free SGD competitive, and (b) apply first-order momentum only to the output layer where gradient variance is highest. Combined: SCALE.
- **Why it might help here:** The "momentum only on the output layer" insight is unusually surgical and cheap to test — and record 18's lm_head currently uses the same Adam betas as embeddings. Could be a single-knob improvement.
- **Plug-in surface:** `optimizer1` param groups (separate the lm_head into its own group with tweaked momentum).
- **A100 notes:** Clean.
- **File:** `2506.16659.pdf`

### When, Where, and Why to Average Weights
- **arXiv:** [2502.06761](https://arxiv.org/abs/2502.06761) — 2025-02
- **Key claim:** Systematic study of LAWA-style early weight averaging and EMA. A simple ~1%-of-training averaging window consistently yields optimal results across workloads; appropriately tuned EMA matches LAWA gains.
- **Why it might help here:** At 90 s of training, smoothing the last few steps via EMA before val-eval is a 5-line change with reported "free" gains. Particularly interesting because the validation pass in `train.py` runs *after* training stops — perfect place to swap to averaged weights.
- **Plug-in surface:** Add an EMA module shadowing model params; in the final-eval block, load EMA weights before `val_loader` loop.
- **A100 notes:** Clean. Adds a second set of bf16 weights (~250 MB) — trivially under VRAM budget.
- **File:** `2502.06761.pdf`

## Attention & architecture

### Peri-LN: Revisiting Normalization Layer Placement
- **arXiv:** [2502.02732](https://arxiv.org/abs/2502.02732) — 2025-02
- **Key claim:** Places layer normalization peripherally around each sublayer — i.e. norms both the input *and* the output. Constrains residual spikes that pre-LN suffers from, while keeping pre-LN's gradient pathway. Now adopted in Gemma and OLMo.
- **Why it might help here:** Record 18 is pre-LN (norm before attn/MLP). Adding a peri-LN-style output norm is a 2-line change per block. Direct hit on training stability and final loss.
- **Plug-in surface:** `Block.forward` — add `norm(...)` around each sublayer output.
- **A100 notes:** Clean — F.rms_norm is well-supported.
- **File:** `2502.02732.pdf`

### HybridNorm: QKV-Norm in Attention + Post-Norm in FFN
- **arXiv:** [2503.04598](https://arxiv.org/abs/2503.04598) — 2025-03
- **Key claim:** Hybrid normalization scheme: QKV-norm inside attention (extends QK-norm to V) plus post-norm in the FFN. Reports more stable and efficient transformer pretraining.
- **Why it might help here:** Record 18 already has QK-norm (so half the recipe is in). Adding V-norm and FFN post-norm is a small architectural delta with a defensible mechanistic story.
- **Plug-in surface:** `CausalSelfAttention.forward` (norm `v` after projection) + `Block.forward` (post-norm around MLP output).
- **A100 notes:** Clean.
- **File:** `2503.04598.pdf`

### nGPT: Normalized Transformer on the Hypersphere
- **arXiv:** [2410.01131](https://arxiv.org/abs/2410.01131) — 2024-10
- **Key claim:** All embeddings, MLP, attention, and hidden-state vectors are unit-norm-normalized; each layer is interpreted as a Riemannian step on the hypersphere. Reports 4–20× fewer training steps for the same accuracy.
- **Why it might help here:** A 4–20× sample-efficiency claim is exactly the kind of architectural change worth chasing in a 90-second-budget regime. Even if the real-world factor is much smaller, the upside dominates.
- **Plug-in surface:** Substantial — touches every linear layer, plus learnable per-direction scalars. Largest implementation effort on this list, biggest claimed upside.
- **A100 notes:** Clean — fewer FP precision issues than baseline (normalized vectors are easier to keep in bf16). Seed reference (2024-10), but the idea is still hot in 2025/2026 follow-ups.
- **File:** `2410.01131.pdf`

### DeepCrossAttention: Supercharging Residual Connections
- **arXiv:** [2502.06785](https://arxiv.org/abs/2502.06785) — 2025-02
- **Key claim:** Replaces the additive residual with a learned cross-attention over previous-layer outputs. Reports better training-loss-vs-FLOPs than a vanilla pre-LN residual.
- **Why it might help here:** Record 18 has a U-Net skip pattern + learnable per-layer skip_weights. DeepCrossAttention is a strictly richer skip mechanism — natural extension of the existing skip design.
- **Plug-in surface:** `Block.forward` skip handling + `GPT.forward` U-Net stack.
- **A100 notes:** Clean. Adds a small attention op per residual; VRAM cost modest.
- **File:** `2502.06785.pdf`

### Enhanced QKNorm with Lp Norm
- **arXiv:** [2602.05006](https://arxiv.org/abs/2602.05006) — 2026-02
- **Key claim:** Generalizes QK-norm beyond L2 to general Lp norms. Reports gains in gradient propagation and attention sharpness from non-Euclidean choices.
- **Why it might help here:** A 1-line generalization of a baseline component (record 18's `norm(q), norm(k)` after rotary), parameterized by p. Cheap to test, defensible upside.
- **Plug-in surface:** Replace the L2 `norm` calls on Q, K in `CausalSelfAttention.forward` with a parameterized Lp variant.
- **A100 notes:** Clean.
- **File:** `2602.05006.pdf`

### mHC: Manifold-Constrained Hyper-Connections
- **arXiv:** [2512.24880](https://arxiv.org/abs/2512.24880) — 2025-12
- **Key claim:** Extension of the Hyper-Connections (ICLR 2025) residual-replacement scheme. Projects HC's residual space onto a specific manifold to restore identity mapping, with efficiency optimizations.
- **Why it might help here:** Hyper-connections are the technique behind the upstream record 73 ("partitioned hyperconnections", Feb 2026). They aren't in record 18 yet, so this — plus its mHC refinement — is a high-leverage architectural rewrite candidate.
- **Plug-in surface:** Replace residual addition (`x = x + self.attn(...)` and `x = x + self.mlp(...)`) with an HC-style multi-stream residual scheme.
- **A100 notes:** Mostly clean; per-paper, modest VRAM increase from extra residual streams.
- **File:** `2512.24880.pdf`

## Schedule & curriculum

### Training Dynamics of the Cooldown Stage in Warmup-Stable-Decay
- **arXiv:** [2508.01483](https://arxiv.org/abs/2508.01483) — 2025-08
- **Key claim:** Analyzes cooldown-shape choices in WSD schedules. Finds a fundamental bias–variance trade-off — shapes that balance exploration vs. exploitation consistently outperform alternatives.
- **Why it might help here:** Record 18's schedule is linear decay starting at `cooldown_frac = 0.4`. The paper provides principled cooldown shapes (specific curves rather than "linear or cosine") that are 5-line drop-ins for `get_lr`.
- **Plug-in surface:** `get_lr` function in `train.py`.
- **A100 notes:** Clean.
- **File:** `2508.01483.pdf`

### How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining
- **arXiv:** [2511.18903](https://arxiv.org/abs/2511.18903) — 2025-11
- **Key claim:** Aggressive LR decay underuses the best (later) data. Moderate decay holding higher LR during the high-quality phase outperforms cosine, especially with weight averaging.
- **Why it might help here:** Record 18 decays LR all the way down to `0.15` of base over the last 40% of training. A less-aggressive cooldown (combined with EMA from `2502.06761`) is plausibly free 90-s val_loss.
- **Plug-in surface:** `get_lr` decay floor + (optional) EMA in eval.
- **A100 notes:** Clean.
- **File:** `2511.18903.pdf`

### Seesaw: Accelerating Training by Balancing LR and Batch-Size Scheduling
- **arXiv:** [2510.14717](https://arxiv.org/abs/2510.14717) — 2025-10
- **Key claim:** Joint LR/batch-size schedule that "trades" between them over training. Reports speedups vs. independently tuned LR/BS schedules.
- **Why it might help here:** Record 18 has a fixed batch size (8 × 64K tokens). Upstream record 46 introduced a batch-size schedule that helped; Seesaw is the published analysis of the same family of ideas — perfect place to consult before reinventing.
- **Plug-in surface:** `args.batch_size` (make step-dependent) + `get_lr` (re-tied to BS schedule).
- **A100 notes:** Watch VRAM at the high-BS end of the schedule.
- **File:** `2510.14717.pdf`

### Critical Batch Size Revisited
- **arXiv:** [2505.23971](https://arxiv.org/abs/2505.23971) — 2025-05
- **Key claim:** Simple empirical recipe for finding the critical batch size where wall-clock vs. tokens trade favorably. Re-examines old assumptions about scaling laws and BS.
- **Why it might help here:** Direct calibration question. Record 18 may be over- or under-sized on batch for the A100 setup. The paper's recipe is implementable as a one-line BS sweep guided by gradient-noise-scale.
- **Plug-in surface:** `args.batch_size`.
- **A100 notes:** Clean.
- **File:** `2505.23971.pdf`

### On the Effectiveness of Infinite LR Schedule for Pretraining
- **arXiv:** [2503.02844](https://arxiv.org/abs/2503.02844) — 2025-03
- **Key claim:** WSD-style "stable" phase with no decay (or near-no-decay) extended for the bulk of training, with very late cooldown. Argues this is robust to unknown final horizon.
- **Why it might help here:** At 90 s training, much of the training is "warm-up + cooldown" — actually almost no stable phase. An infinite-LR schedule with a hard cooldown at the end is a non-trivial alternative to record 18's `cooldown_frac=0.4` linear decay.
- **Plug-in surface:** `get_lr` function.
- **A100 notes:** Clean.
- **File:** `2503.02844.pdf`

## Loss & output head

### Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries
- **arXiv:** [2510.14751](https://arxiv.org/abs/2510.14751) — 2025-10
- **Key claim:** MTP captures only short-range dependencies. Future Summary Prediction (FSP) trains an auxiliary head to predict a compact representation of the long-term future. Improves over both NTP and MTP at scale.
- **Why it might help here:** Record 18 has standard NTP (single softmax over next token). Upstream record 53 added MTP — a strong indicator this family of objectives helps. FSP is the next step beyond MTP; either MTP or FSP is plausibly a meaningful 90-s gain.
- **Plug-in surface:** Add an auxiliary head + auxiliary loss term to `GPT.forward`.
- **A100 notes:** Mild VRAM increase from the aux head; modest.
- **File:** `2510.14751.pdf`

### Pre-Training Curriculum for Multi-Token Prediction
- **arXiv:** [2505.22757](https://arxiv.org/abs/2505.22757) — 2025-05
- **Key claim:** Small LMs struggle with the MTP objective. A forward/reverse curriculum that ramps the MTP complexity over training enables smaller models to benefit.
- **Why it might help here:** Critical for our 124M-param scale — vanilla MTP may *hurt* at this size, but curriculum-MTP might still be a win. Worth pairing with the Beyond-MTP paper above.
- **Plug-in surface:** Aux-head + step-dependent loss weighting (curriculum) in the training loop.
- **A100 notes:** Clean.
- **File:** `2505.22757.pdf`

### L-MTP: Leap Multi-Token Prediction
- **arXiv:** [2505.17505](https://arxiv.org/abs/2505.17505) — 2025-05
- **Key claim:** Instead of predicting the next K adjacent tokens, predict K *non-adjacent* (leap) future tokens. Reports better long-range gradient signal than vanilla MTP.
- **Why it might help here:** Variant of MTP that may sidestep the "small models can't predict close tokens reliably" issue raised by the curriculum paper.
- **Plug-in surface:** Same aux-head as MTP, but targets are non-adjacent indices.
- **A100 notes:** Clean.
- **File:** `2505.17505.pdf`

## Numerical & auxiliary regularization

### Cautious Weight Decay (CWD)
- **arXiv:** [2510.12402](https://arxiv.org/abs/2510.12402) — 2025-10
- **Key claim:** One-line, optimizer-agnostic modification: apply weight decay *only* to parameter coordinates whose signs align with the optimizer update. Improves AdamW, Lion, and Muon across 338M / 986M / 2B pretraining.
- **Why it might help here:** Single-line change to Muon's weight-decay step. Strong reported gains, broadly applicable, well within the simplicity criterion in `program.md`. Was added to upstream at record 43 — not yet in record 18.
- **Plug-in surface:** Muon's `p_world.data.add_` and the `optimizer1` Adam group (weight-decay term).
- **A100 notes:** Clean — pure bf16 elementwise.
- **File:** `2510.12402.pdf`

### AlphaDecay: Module-wise Weight Decay for Heavy-Tailed Balancing
- **arXiv:** [2506.14562](https://arxiv.org/abs/2506.14562) — 2025-06
- **Key claim:** Different modules have different gradient heavy-tail characteristics; one-size-fits-all weight decay is suboptimal. Per-module decay rates derived from a heavy-tail diagnostic.
- **Why it might help here:** Record 18 uses two flat WD values (one for Adam, one for Muon). Module-wise WD gives per-block knobs; combined with CWD, the optimizer-side surface area for free-lunch gains expands.
- **Plug-in surface:** Per-parameter-group `weight_decay` in `optimizer1` + Muon's WD.
- **A100 notes:** Clean.
- **File:** `2506.14562.pdf`

### SPECTRA: Enhancing LLM Training via Spectral Clipping
- **arXiv:** [2603.14315](https://arxiv.org/abs/2603.14315) — 2026-03
- **Key claim:** Identifies "sparse spectral spikes" in stochastic gradients — a few singular values much larger than the rest. Pre-clipping singular values + post-clipping update norms uniformly improves AdamW, Signum, AdEMAMix on LM pretraining.
- **Why it might help here:** Muon already orthogonalizes the matrix update (so post-clipping is partially redundant for hidden matrices). But pre-clipping the *gradient* before the momentum buffer is updated is novel relative to record 18, and applies to the Adam-side groups too.
- **Plug-in surface:** Insert spectral-clip op before the momentum lerp in `Muon.step`, and an analogous gradient-norm clamp in `optimizer1`'s loop.
- **A100 notes:** Clean — SVD-based clipping is bf16-friendly with fp32 fallback.
- **File:** `2603.14315.pdf`

---

## How this was curated

I cast a wide net via WebSearch across 9 plug-in surfaces (Muon
variants, alternative optimizers, MTP, hyperconnections / residuals,
RoPE variants, normalization, schedules, batch-size / curriculum, and
weight-decay variants), then triaged by claimed effect magnitude,
implementation simplicity, and overlap with the record-18 baseline.

**Deliberate omissions, so the agent doesn't redo the search:**
- The original Muon paper (Bernstein–Newhouse 2024), the original
  value-residual / ResFormer paper, the original ReLU² paper, the
  original QK-norm paper, the original FlexAttention / sliding-window
  papers, the original Polar Express paper, the original NorMuon paper,
  and the original Hyper-Connections paper — all of these techniques
  are *already in the record-18 baseline or in a record downstream of
  it*. The literature folder lists *new* ideas relative to the baseline.
- Hopper-only techniques (FP8 LM head, FA3, TMA Triton kernels) —
  rejected outright; they don't run on Ampere.
- MoE-specific and 1B+-scale-only papers — rejected; record 18 is a
  dense 124M-param GPT, and the 90-s budget rules out very large models.
- Pure post-training / RLHF / fine-tuning papers — rejected; this is
  a pretraining benchmark.

**Coverage gap to be aware of.** I underweighted "novel attention shapes"
(linear attention, retentive networks, Mamba-style SSMs) because they
require deep replacement of FlexAttention rather than a `train.py`
edit. The agent could explore that direction by reading the upstream
record 80 (paired-head Muon) and record 58 (paired-head attention),
which are both A100-compatible and listed in
`../speedrun_original/README.md`.
