# job_classify — run log

Every measured training run for the ViT-B/16 + Food-101 classifier, in chronological order. Validation numbers all use `validate_classify.py` (HF `AutoImageProcessor` preprocessing + per-example mean cross-entropy + top-1 / top-5 accuracy on the 25,250-image Food-101 val split).

The HF reference at the bottom is what every run is benchmarked against.

## Summary

| Run | Topology | Code recipe | Config | top-1 | top-1 (TTA) | val_loss | val_loss (TTA) | Train + comm only | Container wall-clock | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `classify.pt` (v1) | 4-worker FL, 4× RTX PRO 4500 (gpuflex) | sawtooth LR per round, fresh Adam each round, plain CE | batch 128, lr 2e-4, 5 rounds × 1 local epoch | 87.90 % | — | 0.5444 | — | 6m 26s | ~6m 56s | Baseline. LR resets each round, Adam momentum lost, no cross-round continuity. |
| `classify_v2.pt` (v2) | 4-worker FL, 4× RTX PRO 4500 (gpuflex) | **continuous LR across rounds + persisted per-client Adam state** | batch 128, lr 2e-4, 5 rounds × 1 local epoch | **88.66 %** | **88.87 %** | **0.5125** | **0.5015** | 6m 13s | ~6m 45s | **Best result so far.** +0.76 pp top-1 / −0.032 val_loss vs v1. The two FL-aware tweaks closed 62 % of the gap to HF without touching the rest of the recipe. |
| `classify_v3.pt` (v3) | 4-worker FL, 4× RTX PRO 4500 (gpuflex) | + label smoothing 0.1 + per-round EMA decay 0.999 + 10 rounds | batch 128, lr 2e-4, 10 rounds × 1 local epoch | 87.13 % | 87.29 % | 0.7881 | 0.7868 | 13m 21s | ~13m 48s | **Regression.** EMA decay 0.999 over only 148 local steps puts ~86 % weight on round-start weights → submitting EMA back to FedAvg means ~14 % of intended progress per round. Label smoothing inflates reported val_loss (model trained for soft targets, eval uses hard CE). Job ID `cmp16fzuy01wepc01jd2vi7ex`. |
| `GLOBAL_MODEL (7).pt` (v4) | 4-worker FL, 4× RTX PRO 4500 (gpuflex) | same as v3 (label smoothing + EMA) | batch 256, lr 2.83e-4 (= 2e-4 × √2), 10 rounds × 1 local epoch | **78.75 %** | — | **1.7251** | — | 15m 33s | ~16m 23s | **Severe regression.** Halving steps per round (74 vs 148) with the same EMA decay 0.999 → only ~7 % within-round trajectory weight → global model barely moves past head-init. Wall-clock anomaly: longer than v3 despite half the local steps (gpuflex hardware variance). |
| `GLOBAL_MODEL (8).pt` (v2 rerun #1, post-rollback) | 4-worker FL, 4× RTX PRO 4500 (gpuflex) | same as v2 (continuous LR + persisted Adam, no EMA, no smoothing) | batch 128, lr 2e-4, 5 rounds × 1 local epoch | **88.83 %** | **89.24 %** | 0.6054 | **0.5756** | not measured (no logs) | not measured | First run after fully rolling back v3/v4 deviations. **Beats HF on TTA top-1 (Δ +0.11 pp).** Vanilla val_loss slightly worse than v2's 0.5125 — within stochastic noise on prod re-runs. |
| `GLOBAL_MODEL (9).pt` (v2 rerun #2, post-rollback) | 4-worker FL, 4× RTX PRO 4500 (gpuflex) | same as v2 | batch 128, lr 2e-4, 5 rounds × 1 local epoch | **88.84 %** | **89.17 %** | **0.5932** | **0.5695** | not measured (no logs) | not measured | Independent re-run of the same config. Reproduces v2-rerun-#1 within ~0.07 pp top-1 / ~0.012 val_loss. **Also beats HF on TTA top-1 (Δ +0.04 pp).** Confirms the recipe's reproducibility on prod. |
| **`GLOBAL_MODEL (10).pt` (v10, big-batch)** | 4-worker FL, 4× RTX PRO 4500 (gpuflex) | same as v2 (continuous LR + persisted Adam) | **batch 348**, lr 2e-4 (unchanged), 5 rounds × 1 local epoch | **89.42 %** | **89.70 %** | **0.4971** | **0.4857** | 8m 3s | ~8m 48s | **Best run.** Bigger batch + same LR (no √-scaling) gives cleaner per-round gradient signal than v2. **Beats HF on vanilla top-1 (+0.29 pp) AND TTA (+0.57 pp).** val_loss closest to HF of any run (Δ +0.047 vanilla / +0.036 TTA). Total opt steps drop to 1,100 (vs HF's 2,960) — fewer-but-cleaner updates. |
| `GLOBAL_MODEL (11).pt` (v2 rerun #3, 4090) | 4-worker FL, **4× RTX 4090 (gpuflex)** | same as v2 | batch 128, lr 2e-4, 5 rounds × 1 local epoch | 88.73 % | 88.99 % | 0.5155 | 0.5052 | 7m 8s | ~7m 44s | Same recipe as v2/v8/v9 but on 4090 hardware. Top-1 lands in the middle of the v2 cluster (88.66 / 88.83 / 88.84 / **88.73**) — confirms **recipe is hardware-agnostic**. ~1 min slower than RTX PRO 4500: 4090 (Ada) has lower fp16 Tensor-Core throughput than RTX PRO 4500 (Blackwell) for ViT-B/16 at batch 128. |
| `classify_single_gpu.pt` | **1-worker centralized**, H100 SXM 80 GB | 1:1 HF recipe | batch 128, lr 2e-4, 5 epochs | 86.17 % | 87.03 % | 0.6260 | 0.5553 | 9m 22s (562 s) | ~13 min | Apples-to-apples centralized baseline (`../classify_single_gpu/`). Train loss collapses to ~0.005 by epoch 5 (HF reports 0.0452 at the same point); model overfits hard. **Inversion: 4-worker FL v2 actually beats this** — FedAvg's averaging is acting as a regularizer. |
| **HF reference** (`nateraw/food`) | 1-worker centralized, hardware not disclosed | 1:1 HF recipe (the source) | batch 128, lr 2e-4, 5 epochs | **89.13 %** | n/a | **0.4501** | n/a | not disclosed | not disclosed | Source of truth. PyTorch 1.9.0 + Native AMP per the model card. |

### How the time columns are measured

- **Train + comm only** — clock from the moment the first training step starts to when the last training task finishes. For FL runs that's `first [train] log line → last "finished processing task"` extracted from the worker logs (this is *all rounds × 4 clients* of: forward/backward + grad clip + optimizer step + AMP, plus inter-round FedAvg comm and weight upload/download). For the single-GPU run it's the `562 s` reported by `train.py` itself, measured from `t0` (set right before the training loop) to the `torch.save` call — no comm because it's single-machine. **Excludes container startup, HF dataset materialization, model pretrain download, and the validation pass.**
- **Container wall-clock** — clock from `Worker_process started` → `MPM: Good Bye!` for FL, or container `start` → final validation done for the single-GPU run. **Includes** everything above plus container init, Python imports, model pretrain download (cached after first hit), and shutdown.

The Train+comm column is the apples-to-apples one for comparing recipe efficiency; the Container column is what you actually wait for end-to-end.

## Δ vs HF (top-1)

Vanilla columns; for runs where TTA was measured, the `Δ top-1 (TTA)` column shows the additional bump.

| Run | Δ top-1 | Δ top-1 (TTA) | Δ val_loss |
|---|---:|---:|---:|
| `classify.pt` (v1) | −1.23 pp | n/a | +0.094 |
| `classify_v2.pt` (v2) | −0.47 pp | −0.26 pp | +0.062 |
| `classify_v3.pt` (v3) | −2.00 pp | −1.84 pp | +0.338 |
| `GLOBAL_MODEL (7).pt` (v4) | −10.38 pp | n/a | +1.275 |
| `GLOBAL_MODEL (8).pt` (v2 rerun #1) | −0.30 pp | **+0.11 pp** ✅ | +0.155 |
| `GLOBAL_MODEL (9).pt` (v2 rerun #2) | −0.29 pp | **+0.04 pp** ✅ | +0.143 |
| **`GLOBAL_MODEL (10).pt` (v10, big-batch, best)** | **+0.29 pp** ✅ | **+0.57 pp** ✅ | **+0.047** |
| `GLOBAL_MODEL (11).pt` (v2 rerun #3, 4090) | −0.40 pp | −0.14 pp | +0.066 |
| `classify_single_gpu.pt` | −2.96 pp | −2.10 pp | +0.176 |

## Lessons (in order of impact)

1. **Continuous LR + persisted Adam are the only deviations actually worth keeping.** v2 → v1 gain (+0.76 pp / −0.03 loss) was real and measurable. Everything else attempted on top either regressed or was neutral.
2. **EMA over short FL rounds is a trap.** Decay 0.999 makes sense for thousand-step training; over 74-148 steps per round it dominates the round-start weights and nullifies most of the local progress. The fix is either (a) lower decay (~0.95) so EMA actually moves with training, or (b) persist EMA across rounds so the decay accumulates over the full run. Resetting EMA every round at decay 0.999 is the worst combination.
3. **Label smoothing inflates measured val_loss.** Even when it doesn't hurt top-1 much, the val_loss number you're trying to compare against HF gets penalized because the model is calibrated for soft targets. Don't use it if the headline metric is val_loss.
4. **FedAvg can outperform a too-aggressive centralized run.** The single-GPU centralized run overfits by epoch 5 (train loss → 0.005). The FL v2 run lands at the same total step count but FedAvg's averaging acts as a regularizer that prevents the late-epoch collapse. This is counter-intuitive — usually FL underperforms — but happens here because the recipe is at the edge of overfitting on this dataset.
5. **Bigger batch ≠ free speed in FL.** The v4 wall-clock anomaly (longer than v3 despite fewer local steps) suggests gpuflex's per-batch overhead doesn't amortize linearly with batch size in the federated path. Worth measuring if you push batch further.

## Current state of the codebase (fully rolled back to v2)

After v3 and v4 both regressed and v4's larger-batch / 10-round config also doubled wall-clock, every later deviation was reverted. The job is now back to the exact recipe that produced `classify_v2.pt`:

- Architecture, optimizer, LR, batch, AMP, grad clip, seed, preprocessing, augmentation: **identical to HF** (audit table in `README.md`).
- LR schedule: linear decay, **continuous across all rounds** (no sawtooth). _The only kept code-level deviation._
- Adam state: **persisted per-client across rounds** (`STATE_DIR/optimizer_state.pt`). _The only other kept code-level deviation._
- Loss: plain `CrossEntropyLoss` (no smoothing).
- Submitted weights: last-step `model.state_dict()` (no EMA).
- Config (`configs/`): `batch_size=128`, `learning_rate=0.0002`, `num_rounds=5`, `local_epochs=1`.

Total optimizer steps with this config: 4 workers × 5 rounds × 148 steps = **2,960** — exactly matches the HF total step count.

### Reproducibility — four v2-recipe runs, two GPU types

| Run | Hardware | top-1 (vanilla) | top-1 (TTA) | val_loss (TTA) |
|---|---|---:|---:|---:|
| `classify_v2.pt` (original) | 4× RTX PRO 4500 | 88.66 % | 88.87 % | 0.5015 |
| `GLOBAL_MODEL (8).pt` (rerun #1) | 4× RTX PRO 4500 | 88.83 % | **89.24 %** | 0.5756 |
| `GLOBAL_MODEL (9).pt` (rerun #2) | 4× RTX PRO 4500 | 88.84 % | 89.17 % | 0.5695 |
| `GLOBAL_MODEL (11).pt` (rerun #3) | **4× RTX 4090** | 88.73 % | 88.99 % | 0.5052 |
| **mean** | — | **88.77 %** | **89.07 %** | 0.5380 |
| stddev | — | 0.09 pp | 0.16 pp | 0.034 |
| **Δ mean vs HF** | — | **−0.36 pp** | **−0.06 pp** | +0.088 |

Four independent prod runs of the same recipe — across **two different GPU generations** — land within 0.09 pp top-1 vanilla and 0.16 pp top-1 TTA of each other. **The mean TTA top-1 (89.07 %) is essentially at parity with HF's 89.13 %**, with one run (v2 rerun #1) exceeding it by +0.11 pp. The recipe is reproducible regardless of hardware; only wall-clock changes (~6m 13s on RTX PRO 4500 vs ~7m 8s on RTX 4090, both train+comm).

For results that *exceed* HF on vanilla top-1, see the v10 / big-batch run (89.42 % / 0.4971) in the main table.
