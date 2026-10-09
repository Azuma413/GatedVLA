# QwenDrivePI on PAI-AV (front camera)

Qwen-Drive-1.0-4B VLM + pi-style flow-matching action expert (`QwenDrivePI`), trained on the raw
`nvidia/PhysicalAI-Autonomous-Vehicles` release (front wide camera + egomotion only, no LiDAR / radar).

| | |
|---|---|
| image | front wide camera at t0 - 0.1 s (544x288) and t0 (1280x704) |
| state (5, 10) | ego history t0-0.4 .. t0 @10 Hz: speed, accel xyz, ang. vel xyz (ego frame), heading rel. to t0, target point xy |
| action (10, 2) | ego-frame (x fwd, y left) positions t0 .. t0+0.9 s @10 Hz |
| target point | first future ego position >= 25 m away (PAI-AV has no route) |
| normalisation | per-dim q01/q99 -> [-1, 1] (`norm_stats.json`) |

Code: `starVLA/dataloader/pai_av_drive.py` (dataset), `starVLA/training/train_drive.py` (trainer),
`starVLA/training/drive_openloop.py` (metrics + BEV plots), `starVLA/model/framework/VLM4A/QwenDrivePI.py`.

## 1. Prepare (once, login node is fine)

```bash
module use /soft/modulefiles && module load frameworks
source ../../outputs/starvla_venv/bin/activate          # from lib/starVLA
python examples/drive/PAI-AV/prepare_paiav.py \
  --data_root /lus/flare/projects/RobustViT/hiratsuka/datasets/PhysicalAI-Autonomous-Vehicles \
  --out_dir ../../outputs/qwendrive_pi/meta
```

Writes `norm_stats.json` (6000 train clips x 8 t0) and `eval_samples_val.json`
(240 fixed val samples: 80 left / 80 right / 80 straight, by heading change over the next 3 s, +-15 deg).

## 2. Train (Aurora, 1 node = 12 XPU tiles)

```bash
cd lib/starVLA
# smoke test (debug queue)
qsub -v MAX_STEPS=20,EVAL_INTERVAL=10,RUN_ID=qdpi_smoke examples/drive/PAI-AV/train_files/train_qwendrive_pi_aurora.pbs
# full run; re-submit the same command to resume (optimizer state + same W&B run)
qsub -q capacity -l walltime=24:00:00 -v TIME_LIMIT_HOURS=23.3 examples/drive/PAI-AV/train_files/train_qwendrive_pi_aurora.pbs
```

Live log: `logs/<RUN_ID>.log`. Outputs in `outputs/qwendrive_pi/<RUN_ID>/`:
`checkpoints/step_*/model.safetensors` (bf16 weights, last 2), `resume/` (sharded model + optimizer, latest only),
`eval/step_*/*.png` (BEV plots), `wandb/`.

## 3. What is logged to W&B (project `starVLA_drive`)

- `train/action_loss`, `train/grad_norm`, `train/lr_*`, `train/samples_per_s`, `train/data_wait_frac` every `logging_frequency` steps.
- every `eval_interval` steps (and at step 0), open-loop on the fixed val set:
  `eval/ADE`, `eval/FDE`, `eval/L2@0.5s`, `eval/L2@1.0s`, `eval/lat_err_final`, `eval/lon_err_final`, `eval/miss_rate_2m`,
  each also per maneuver (`eval/left/*`, `eval/right/*`, `eval/straight/*`).
- `eval_bev/{left,right,straight}`: front image + bird's-eye view with past (2 s), GT future (5 s, dashed),
  GT horizon, prediction and target point, for the same samples at every eval.

## 4. Visual evidence gate (EyePolicy) — `starvla_qwendrive_pi_paiav_eye.yaml`

Port of `lib/EyePolicy` (π0.5 `eye` policy) to the layer-wise DiT expert:

| | EyePolicy (π0.5) | QwenDrivePI |
|---|---|---|
| gate input | final-layer hidden of each image token `h_p`, mean text hidden `q` | same (Qwen3.5 final layer; `q` = mean over all non-visual, non-pad tokens) |
| gate head | per camera `W2·swish(W1·[h_p, q, h_p⊙q])`, RMSNorm, init keep 0.99, τ = 1, no floor | same, one head shared by both frames (same camera); budget still per frame |
| where it acts | `log m` on action-expert → image-token attention logits, every layer | `log m` on DiT cross-attention logits to visual tokens, every cross-attention block |
| text-token path | not blocked | not blocked |
| loss | `L_gated + λ_full·L_full + λ_b·Σ_cam (mean m − ρ)²`, shared noise / t | same (ρ = 0.2, λ_b = 0.1, λ_full = 1) |
| controls | `uniform` (m = ρ), `random` (Bernoulli(ρ)) | same (`framework.eye_gate.mode`) |

Logged: `train/{loss_gated,loss_full,loss_budget,keep_view0,keep_view1}`, `eval/keep_view*`, and a "gated view"
panel (current frame × keep map) in `eval_bev/*`.

Planned combination with AnchorVLA (not implemented): the anchor classifier would read a learned query token
(query-based DAAR) and the DiT would generate the residual to the selected anchor (DARF). The gate's context `q`
is the hook for the combination (e.g. `[q, anchor embedding]`), so the gate can depend on the chosen maneuver.
Note the current 0.9 s horizon is likely too short for maneuver-level anchors.

## 5. Decision anchors (AnchorVLA, modified) — `starvla_qwendrive_pi_paiav_anchor_*_h25.yaml`

- Codebook: `prepare_paiav.py --action_horizon 26 --num_anchors 100` -> `meta_h25_anchor/anchors.npz`
  (k-means of 2.5 s train trajectories in metres; per-step std of the residual to the nearest anchor).
- DAAR (query-based): `<|quad_start|>` is appended to the prompt and its embedding replaced by a learned query;
  an MLP on its final hidden gives K logits, trained with the soft target over the 8 nearest anchors (γ = 1 m).
- Reference: softmax-weighted mean of the top-6 anchors (differentiable; `reference: hard` = top-1, no gradient).
- DARF: the DiT generates `(τ − reference) / σ_r` (L1 flow loss); the reference is a per-step query bias in
  every DiT block (query only). No top-M candidates / confidence branch.
- Stage 1 (6 epochs, `anchor_s1`): VLM + query + classifier; the flow expert is frozen. Prediction = reference.
- Stage 2 (12 epochs, `anchor_s2` / `anchor_eye_s2`): from the stage-1 weights, classifier + residual flow
  (+ eye gate). `trainer.pretrained_checkpoint` points to the stage-1 final checkpoint (`step_0019200`).
- Eval: `eval/anchor/*` — top-1 accuracy / top-6 recall vs. the GT-nearest anchor, how mixed the top-6 are
  (`topn_mixed_class_frac`, `topn_left_and_right_frac`, `topn_heading_spread_deg`), softmax weight entropy, and
  the ADE/FDE of the reference alone. BEV plots show the top-6 anchors and the reference.
