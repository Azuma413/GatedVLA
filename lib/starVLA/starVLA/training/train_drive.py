# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Train a driving VLA (QwenDrivePI) on PAI-AV with W&B logging and periodic open-loop evaluation.

Launch modes
    single process   : python starVLA/training/train_drive.py --config <yaml> [key=value ...]
    torchrun (CUDA)  : torchrun --nproc_per_node N starVLA/training/train_drive.py --config <yaml>
    Aurora (XPU)     : mpiexec --pmi=pmix --envall -n 12 -ppn 12 python -u starVLA/training/train_drive.py ...
                       (rank / tile pinning + xccl process group are set up from MPI before any heavy import)

Multi-rank runs shard parameters with FSDP2 (fp32 master weights, bf16 compute, fp32 gradient reduction).
Every ``trainer.eval_interval`` optimizer steps the model is evaluated open-loop on a fixed, maneuver-balanced
set of val samples: displacement metrics (overall and per left/right/straight) and BEV plots of predicted vs.
ground-truth trajectories are logged to W&B.
"""

import argparse
import json
import math
import os
import shutil
import socket
import sys
import time
import uuid
from contextlib import nullcontext
from pathlib import Path


def setup_distributed():
    """Pin this rank's device and create the process group. Returns (rank, world, local_rank, device)."""
    import torch

    launched_by_mpi = "RANK" not in os.environ and any(
        k in os.environ for k in ("PALS_LOCAL_RANKID", "PMI_RANK", "OMPI_COMM_WORLD_RANK")
    )
    if launched_by_mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank, world = comm.Get_rank(), comm.Get_size()
        local = int(
            os.environ.get(
                "PALS_LOCAL_RANKID",
                os.environ.get("MPI_LOCALRANKID", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank)),
            )
        )
        master = comm.bcast(socket.gethostname() if rank == 0 else None, root=0)
        # Aurora: rendezvous over the high-speed network interface.
        suffix = os.environ.get("MASTER_ADDR_SUFFIX", ".hsn.cm.aurora.alcf.anl.gov" if "PALS_LOCAL_RANKID" in os.environ else "")
        os.environ["MASTER_ADDR"] = master + suffix
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(local))
    else:
        rank = int(os.environ.get("RANK", 0))
        world = int(os.environ.get("WORLD_SIZE", 1))
        local = int(os.environ.get("LOCAL_RANK", 0))

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.set_device(local)
        device, backend = torch.device(f"xpu:{local}"), "xccl"
    elif torch.cuda.is_available():
        torch.cuda.set_device(local)
        device, backend = torch.device(f"cuda:{local}"), "nccl"
    else:
        device, backend = torch.device("cpu"), "gloo"

    if world > 1 and not torch.distributed.is_initialized():
        kwargs = {} if device.type == "cpu" else {"device_id": device}
        torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world, **kwargs)
    return rank, world, local, device


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. trainer.max_train_steps=10")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    rank, world, local_rank, device = setup_distributed()
    print(f"[rank {rank}/{world}] device {device} ready in {time.time() - t0:.1f}s", flush=True)

    # Heavy imports only after the device is pinned (Aurora: otherwise every rank grabs tile 0).
    import numpy as np
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from safetensors.torch import save_file
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    from starVLA.dataloader.pai_av_drive import DriveSampler, Normalizer, PAIAVDriveDataset, collate_examples
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.training.drive_openloop import (
        MANEUVERS, anchor_metrics, displacement_metrics, fig_to_image, gate_diagnostics, plot_bev, turn_metrics,
    )

    is_main = rank == 0
    distributed = world > 1

    def log(msg):
        if is_main:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def barrier():
        if distributed:
            dist.barrier()

    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    tcfg, dcfg = cfg.trainer, cfg.datasets.vla_data
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    out_dir = Path(cfg.run_root_dir) / cfg.run_id
    if is_main:
        (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        shutil.copy(dcfg.stats_path, out_dir / "norm_stats.json")
    barrier()

    # ── data ──────────────────────────────────────────────────────────
    fw = cfg.framework
    normalizer = Normalizer.from_file(dcfg.stats_path)
    sampler_cfg = DriveSampler(
        state_horizon=int(fw.action_model.get("state_horizon", 5)),
        action_horizon=int(fw.action_model.get("action_horizon", 10)),
        target_distance=float(dcfg.get("target_distance", 25.0)),
        state_mode=str(dcfg.get("state_mode", "drivemoe")),
        action_mode=str(dcfg.get("action_mode", "waypoints")),
        state_target_point=bool(dcfg.get("state_target_point", True)),
    )
    if fw.get("anchor", {}).get("enabled", False):
        anchor_file = np.load(fw.anchor.path)
        anchor_space = str(anchor_file["space"]) if "space" in anchor_file else "waypoints"
        if (anchor_space == "control") != (sampler_cfg.action_mode == "unicycle"):
            raise ValueError(f"{anchor_space} anchors do not match datasets.vla_data.action_mode {sampler_cfg.action_mode}")
    image_kwargs = dict(
        current_image_pixels=int(fw.get("drive", {}).get("current_image_pixels", 921600)),
        history_image_pixels=int(fw.get("drive", {}).get("history_image_pixels", 174080)),
        # navigation target as text in the VLM prompt (the VLM otherwise never sees where to go)
        prompt=str(fw.get("drive", {}).get("prompt", "predict trajectory")),
        target_point_in_prompt=bool(dcfg.get("target_point_in_prompt", False)),
        speed_in_prompt=bool(dcfg.get("speed_in_prompt", False)),
    )
    train_ds = PAIAVDriveDataset(
        dcfg.data_root, normalizer, split="train", sampler=sampler_cfg, max_clips=dcfg.get("max_clips", None),
        **image_kwargs,
    )
    with open(dcfg.eval_index_path) as f:
        eval_samples = json.load(f)[: int(dcfg.get("num_eval_samples", 240))]
    eval_ds = PAIAVDriveDataset(dcfg.data_root, normalizer, samples=eval_samples, sampler=sampler_cfg, **image_kwargs)
    # Figure-only scenes (traffic lights / stop signs / intersections, see prepare_scene_index.py): plotted at every
    # eval, not part of the metrics.
    scene_ds = None
    if dcfg.get("scene_index_path"):
        with open(dcfg.scene_index_path) as f:
            scene_ds = PAIAVDriveDataset(dcfg.data_root, normalizer, samples=json.load(f), sampler=sampler_cfg,
                                         **image_kwargs)
    log(f"train clips {len(train_ds)}, eval samples {len(eval_ds)}, world {world}, device {device}")

    bs = int(dcfg.per_device_batch_size)
    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=seed, drop_last=True)
    num_workers = int(dcfg.get("num_workers", 8))
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_examples,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=int(dcfg.get("prefetch_factor", 4)) if num_workers > 0 else None,
        multiprocessing_context=dcfg.get("worker_start_method", "spawn") if num_workers > 0 else None,
    )

    # ── model ─────────────────────────────────────────────────────────
    log("building model ...")
    model = build_framework(cfg).float()
    log(f"model built ({time.time() - t0:.0f}s since start)")
    if tcfg.get("pretrained_checkpoint"):
        load_weights(model, tcfg.pretrained_checkpoint, log)
    vlm = model.qwen_vl_interface.model
    vlm.config.use_cache = False
    if tcfg.get("gradient_checkpointing", True):
        vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    freeze = [m.strip() for m in str(tcfg.get("freeze_modules", "") or "").split(",") if m.strip()]
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in freeze):
            param.requires_grad_(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f} B, trainable {n_train / 1e9:.2f} B")

    use_bf16 = bool(tcfg.get("bf16", True)) and device.type != "cpu"
    if distributed:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16 if use_bf16 else torch.float32, reduce_dtype=torch.float32)
        units = list(vlm.model.language_model.layers) + list(vlm.model.visual.blocks)
        units += list(model.action_model.model.transformer_blocks)
        for unit in units:
            fully_shard(unit, mp_policy=mp)
        fully_shard(model, mp_policy=mp)
        autocast = nullcontext
        log(f"FSDP2 sharding done ({time.time() - t0:.0f}s since start)")
    else:
        model.to(device)
        autocast = (lambda: torch.autocast(device.type, dtype=torch.bfloat16)) if use_bf16 else nullcontext

    # ── optimizer / schedule ──────────────────────────────────────────
    lr_cfg = tcfg.learning_rate
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        top = name.split(".")[0]
        lr = float(lr_cfg.get(top, lr_cfg.base))
        groups.setdefault((top, lr), []).append(param)
    param_groups = [{"params": ps, "lr": lr, "name": top} for (top, lr), ps in groups.items()]
    opt_cfg = tcfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=tuple(opt_cfg.get("betas", (0.9, 0.95))),
        eps=float(opt_cfg.get("eps", 1e-8)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )
    accum = int(tcfg.get("gradient_accumulation_steps", 1))
    steps_per_epoch = len(train_loader) // accum
    if tcfg.get("num_epochs"):  # epoch budget wins over max_train_steps
        max_steps = int(float(tcfg.num_epochs) * steps_per_epoch)
    else:
        max_steps = int(tcfg.max_train_steps)
    log(f"{steps_per_epoch} optimizer steps per epoch, max_steps {max_steps} ({max_steps / steps_per_epoch:.2f} epochs)")
    warmup = int(tcfg.get("num_warmup_steps", 0))
    min_lr = float(tcfg.get("scheduler_specific_kwargs", {}).get("min_lr", 0.0))

    def lr_lambda_for(base_lr):
        min_ratio = min_lr / base_lr if base_lr > 0 else 0.0

        def f(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        return f

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lr_lambda_for(g["lr"]) for g in param_groups])
    params = [p for g in param_groups for p in g["params"]]
    dit_params = [p for name, p in model.named_parameters() if p.requires_grad and name.startswith("action_model.")]

    # ── W&B ───────────────────────────────────────────────────────────
    wandb = None
    if is_main and cfg.get("use_wandb", True):
        import wandb

        entity = cfg.get("wandb_entity")
        id_file = out_dir / "wandb_run_id.txt"  # resumed jobs keep logging into the same W&B run
        run_id = id_file.read_text().strip() if id_file.exists() else uuid.uuid4().hex[:8]
        id_file.write_text(run_id)
        init_kwargs = dict(
            project=cfg.get("wandb_project", "starVLA_drive"),
            entity=None if entity in (None, "", "your_wandb_entity") else entity,
            name=cfg.run_id,
            id=run_id,
            dir=str(out_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )
        try:
            wandb.init(**init_kwargs)
        except Exception as e:  # compute nodes can lose the route to api.wandb.ai; never let that stop training
            log(f"W&B init failed ({type(e).__name__}: {e}); logging offline to {out_dir}/wandb "
                f"(upload later from a login node: wandb sync {out_dir}/wandb/offline-run-*)")
            wandb.teardown()  # the failed online attempt keeps the run id registered in the wandb service
            wandb.init(mode="offline", **init_kwargs)
        wandb.define_metric("train/*", step_metric="step")
        wandb.define_metric("eval/*", step_metric="step")
        wandb.define_metric("eval_bev/*", step_metric="step")
        OmegaConf.save(cfg, out_dir / "config.yaml")

    # ── open-loop evaluation ──────────────────────────────────────────
    plots_per_class = int(tcfg.get("num_eval_plots_per_class", 4))
    # Plotted samples: per class, distinct clips spread over the eval list, preferring frames brighter than the
    # median (mask overlays are unreadable on night frames). Brightness of every eval frame is measured once, sharded.
    bright = {i: float(np.asarray(eval_ds[i]["image"][-1].convert("L").resize((64, 36))).mean())
              for i in range(len(eval_ds))[rank::world]}
    if distributed:
        parts = [None] * world
        dist.all_gather_object(parts, bright)
        bright = {k: v for part in parts for k, v in part.items()}
    median = float(np.median(list(bright.values())))
    plot_ids = set()
    for m in MANEUVERS:
        seen, cands = set(), []
        for i, s in enumerate(eval_samples[: len(eval_ds)]):
            if s.get("maneuver") == m and s["clip_id"] not in seen:
                seen.add(s["clip_id"])
                cands.append(i)
        pool = [i for i in cands if bright[i] >= median]
        pool = pool if len(pool) >= plots_per_class else cands
        if pool:
            picks = np.linspace(0, len(pool) - 1, min(plots_per_class, len(pool))).round().astype(int)
            plot_ids.update(pool[k] for k in picks)
    eval_bs = int(dcfg.get("eval_batch_size", bs))
    anchors_np = None
    if fw.get("anchor", {}).get("enabled", False):
        anchors_np = np.load(fw.anchor.path)["anchors"]

    def evaluate(step):
        model.eval()
        t_start = time.time()
        shard = list(range(len(eval_ds)))[rank::world]
        per_rank = math.ceil(len(eval_ds) / world)
        records = []
        for b in range(math.ceil(per_rank / eval_bs)):
            ids = shard[b * eval_bs : (b + 1) * eval_bs]
            real = len(ids)
            if real == 0:  # keep collectives in lockstep across ranks
                ids = [shard[0] if shard else 0]
            items = [eval_ds[i] for i in ids]
            with autocast():
                pred_out = model(items, predict=True)
            pred = pred_out["normalized_actions"][:real]
            keep_maps = pred_out.get("keep_maps", [None] * len(items))
            for j, (i, item, p, kmaps) in enumerate(zip(ids[:real], items, pred, keep_maps)):
                meta = item["meta"]
                rec = {
                    "idx": i,
                    "pred_xy": eval_ds.action_to_xy(p, meta),
                    "gt_xy": meta["action_xy"],
                    "maneuver": meta["maneuver"],
                }
                if kmaps is not None:
                    rec["keep_mean"] = [float(k.mean()) for k in kmaps]
                    rec["keep_cur"] = kmaps[-1].astype(np.float32)  # current frame, for the collapse metrics
                for key in ("score_maps", "attn_maps", "attn_maps_textdrop"):
                    if key in pred_out:
                        rec[key[:-5] + "_cur"] = pred_out[key][j][-1].astype(np.float32)
                for key in ("attn_text_mass", "attn_text_mass_textdrop"):
                    if key in pred_out:
                        rec[key] = float(pred_out[key][j])
                if "normalized_actions_textdrop" in pred_out:
                    rec["pred_xy_textdrop"] = eval_ds.action_to_xy(pred_out["normalized_actions_textdrop"][j], meta)
                for name, acts in pred_out.get("control_actions", {}).items():
                    rec.setdefault("pred_xy_controls", {})[name] = eval_ds.action_to_xy(acts[j], meta)
                for name, preds in pred_out.get("probe_preds", {}).items():
                    rec.setdefault("probe", {})[name] = preds[j].astype(np.float32)
                for name, vals in pred_out.get("diag_scalars", {}).items():
                    rec.setdefault("diag", {})[name] = float(vals[j])
                if "anchor_ids" in pred_out:
                    rec.update(
                        anchor_ids=pred_out["anchor_ids"][j],
                        anchor_weights=pred_out["anchor_weights"][j],
                        anchor_top1=int(pred_out["anchor_top1"][j]),
                        anchor_reference=pred_out["anchor_reference"][j],
                    )
                    if "anchor_xy" in pred_out:  # control-space anchors as trajectories of this sample
                        rec["anchor_xy"] = pred_out["anchor_xy"][j]
                if i in plot_ids:
                    thumb = item["image"][-1].copy()
                    thumb.thumbnail((640, 360))
                    rec.update(meta=meta, image=thumb, keep_map=None if kmaps is None else kmaps[-1])
                records.append(rec)
        scene_records = []
        if scene_ds is not None:
            s_shard = list(range(len(scene_ds)))[rank::world]
            for b in range(math.ceil(math.ceil(len(scene_ds) / world) / eval_bs)):
                ids = s_shard[b * eval_bs : (b + 1) * eval_bs]
                real = len(ids)
                if real == 0:  # keep collectives in lockstep across ranks
                    ids = [s_shard[0] if s_shard else 0]
                items = [scene_ds[i] for i in ids]
                with autocast():
                    pred_out = model(items, predict=True)
                for j, (i, item) in enumerate(zip(ids[:real], items)):
                    thumb = item["image"][-1].copy()
                    thumb.thumbnail((640, 360))
                    rec = {"idx": i, "meta": item["meta"], "image": thumb,
                           "pred_xy": eval_ds.action_to_xy(pred_out["normalized_actions"][j], item["meta"]),
                           "keep_map": pred_out["keep_maps"][j][-1] if "keep_maps" in pred_out else None}
                    for key in ("score_maps", "attn_maps"):
                        if key in pred_out:
                            rec[key[:-5] + "_cur"] = pred_out[key][j][-1].astype(np.float32)
                    scene_records.append(rec)
        if distributed:
            gathered = [None] * world
            dist.all_gather_object(gathered, records)
            records = [r for part in gathered for r in part]
            if scene_ds is not None:
                gathered = [None] * world
                dist.all_gather_object(gathered, scene_records)
                scene_records = [r for part in gathered for r in part]
        model.train()
        if not is_main:
            return
        records.sort(key=lambda r: r["idx"])
        metrics = displacement_metrics(
            np.stack([r["pred_xy"] for r in records]),
            np.stack([r["gt_xy"] for r in records]),
            [r["maneuver"] for r in records],
            dt=sampler_cfg.dt,
        )
        if "anchor_ids" in records[0]:
            metrics.update(anchor_metrics(
                np.stack([r["anchor_xy"] for r in records]) if "anchor_xy" in records[0] else anchors_np,
                np.stack([r["gt_xy"] for r in records]),
                np.stack([r["anchor_ids"] for r in records]),
                np.stack([r["anchor_weights"] for r in records]),
                np.array([r["anchor_top1"] for r in records]),
                np.stack([r["anchor_reference"] for r in records]),
                [r["maneuver"] for r in records],
            ))
        if "keep_mean" in records[0]:
            keep_means = np.array([r["keep_mean"] for r in records])
            for v in range(keep_means.shape[1]):
                metrics[f"keep_view{v}"] = float(keep_means[:, v].mean())
            metrics.update(gate_diagnostics(records))
        if "pred_xy_textdrop" in records[0]:
            td = displacement_metrics(
                np.stack([r["pred_xy_textdrop"] for r in records]), np.stack([r["gt_xy"] for r in records]),
                [r["maneuver"] for r in records], dt=sampler_cfg.dt,
            )
            metrics.update({f"textdrop/{k}": v for k, v in td.items() if "/" not in k})
        metrics.update(turn_metrics(np.stack([r["pred_xy"] for r in records]), np.stack([r["gt_xy"] for r in records]),
                                    [r["maneuver"] for r in records]))
        for name in records[0].get("pred_xy_controls", {}):
            ctl_xy, gt_xy = np.stack([r["pred_xy_controls"][name] for r in records]), np.stack([r["gt_xy"] for r in records])
            mans = [r["maneuver"] for r in records]
            metrics.update({f"{name}/{k}": v for k, v in displacement_metrics(ctl_xy, gt_xy, mans, dt=sampler_cfg.dt).items()})
            metrics.update({f"{name}/{k}": v for k, v in turn_metrics(ctl_xy, gt_xy, mans).items()})
        if "probe" in records[0]:  # detached probes on the DiT inputs: correlation with the ground truth
            gt = np.stack([r["gt_xy"] for r in records])
            heading = np.arctan2(gt[:, -1, 1] - gt[:, -2, 1], gt[:, -1, 0] - gt[:, -2, 0])
            truth = {"final_y": gt[:, -1, 1] / 10.0, "y_1p5s": gt[:, min(15, gt.shape[1] - 1), 1] / 10.0, "heading": heading}
            for name in records[0]["probe"]:
                pr = np.stack([r["probe"][name] for r in records])
                for k, (tname, tv) in enumerate(truth.items()):
                    metrics[f"probe_{name}/corr_{tname}"] = float(np.corrcoef(pr[:, k], tv)[0, 1])
        if "diag" in records[0]:
            for name in records[0]["diag"]:
                metrics[f"diag/{name}"] = float(np.mean([r["diag"][name] for r in records]))
        metrics["eval_time_s"] = time.time() - t_start
        log("eval step {}: ADE {:.3f} FDE {:.3f} (left {:.3f} / right {:.3f} / straight {:.3f})".format(
            step, metrics["ADE"], metrics["FDE"], metrics.get("left/ADE", float("nan")),
            metrics.get("right/ADE", float("nan")), metrics.get("straight/ADE", float("nan"))))
        payload = {"step": step, **{f"eval/{k}": v for k, v in metrics.items()}}
        fig_dir = out_dir / "eval" / f"step_{step:07d}"
        fig_dir.mkdir(parents=True, exist_ok=True)
        # every eval sample's predicted / ground-truth positions (for offline analysis)
        np.savez(
            fig_dir / "predictions.npz",
            idx=np.array([r["idx"] for r in records]), maneuver=np.array([r["maneuver"] for r in records]),
            pred_xy=np.stack([r["pred_xy"] for r in records]), gt_xy=np.stack([r["gt_xy"] for r in records]),
            **{f"pred_xy_{name}": np.stack([r["pred_xy_controls"][name] for r in records])
               for name in records[0].get("pred_xy_controls", {})},
            **{f"probe_{name}": np.stack([r["probe"][name] for r in records]) for name in records[0].get("probe", {})},
            **{f"diag_{name}": np.array([r["diag"][name] for r in records]) for name in records[0].get("diag", {})},
        )
        import matplotlib.pyplot as plt

        images = {m: [] for m in MANEUVERS}
        for r in records:
            if "meta" not in r:
                continue
            meta = r["meta"]
            title = f"{r['maneuver']} | {meta['clip_id'][:8]} t0={meta['t0'] * 1e-6:.1f}s | step {step}"
            extra_maps = [(name, r[key]) for name, key in (("gate score z", "score_cur"),
                                                           ("DiT attention (gated, all tokens)", "attn_cur"),
                                                           ("DiT attention (gated, text hidden)", "attn_textdrop_cur"))
                          if key in r]
            if extra_maps:
                np.savez(fig_dir / f"{r['maneuver']}_{r['idx']:04d}.npz", **{k: r[k] for k in (
                    "keep_cur", "score_cur", "attn_cur", "attn_textdrop_cur", "attn_text_mass", "attn_text_mass_textdrop")
                    if k in r})
            fig = plot_bev(
                meta, r["pred_xy"], image=r["image"], title=title, keep_map=r.get("keep_map"), extra_maps=extra_maps,
                reference=r.get("anchor_reference"),
                candidate_anchors=(r["anchor_xy"] if "anchor_xy" in r else anchors_np)[r["anchor_ids"]]
                if "anchor_ids" in r else None,
            )
            image = fig_to_image(fig)
            plt.close(fig)
            image.save(fig_dir / f"{r['maneuver']}_{r['idx']:04d}.png")
            if wandb is not None:
                images[r["maneuver"]].append(wandb.Image(image, caption=title))
        scene_images = []
        if scene_records:
            scene_dir = fig_dir / "scenes"
            scene_dir.mkdir(exist_ok=True)
            for r in sorted(scene_records, key=lambda x: x["idx"]):
                sample = scene_ds.samples[r["idx"]]
                title = f"{sample['caption'][:90]} | {r['meta']['clip_id'][:8]} t0={r['meta']['t0'] * 1e-6:.1f}s | step {step}"
                extra_maps = [(name, r[key]) for name, key in (("gate score z", "score_cur"),
                                                               ("DiT attention (gated, all tokens)", "attn_cur")) if key in r]
                fig = plot_bev(r["meta"], r["pred_xy"], image=r["image"], title=title, keep_map=r["keep_map"],
                               extra_maps=extra_maps)
                image = fig_to_image(fig)
                plt.close(fig)
                image.save(scene_dir / f"scene_{r['idx']:03d}.png")
                scene_images.append(wandb.Image(image, caption=title) if wandb is not None else None)
        if wandb is not None:
            payload.update({f"eval_bev/{m}": imgs for m, imgs in images.items() if imgs})
            if scene_images:
                payload["eval_scenes"] = scene_images
            wandb.log(payload)

    # ── checkpointing ─────────────────────────────────────────────────
    keep_last = int(tcfg.get("keep_last_checkpoints", 2))
    save_dtype = getattr(torch, str(tcfg.get("save_dtype", "bfloat16")))

    def save_checkpoint(step):
        if distributed:
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

            state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        else:
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if is_main:
            if getattr(vlm.config, "tie_word_embeddings", False):
                state = {k: v for k, v in state.items() if not k.endswith("lm_head.weight")}
            state = {k: v.to(save_dtype).contiguous() for k, v in state.items()}
            ckpt = out_dir / "checkpoints" / f"step_{step:07d}"
            ckpt.mkdir(parents=True, exist_ok=True)
            save_file(state, str(ckpt / "model.safetensors"))
            OmegaConf.save(cfg, ckpt / "config.yaml")
            shutil.copy(out_dir / "norm_stats.json", ckpt / "norm_stats.json")
            old = sorted((out_dir / "checkpoints").glob("step_*"))[:-keep_last] if keep_last > 0 else []
            for path in old:
                shutil.rmtree(path, ignore_errors=True)
            log(f"saved {ckpt}")
        barrier()

    # ── resumable training state (sharded model + optimizer, scheduler, step) ──
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import DefaultLoadPlanner
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    resume_root = out_dir / "resume"

    def save_resume_state(step, epoch):
        path = resume_root / f"step_{step:07d}"
        model_sd, optim_sd = get_state_dict(model, optimizer)
        dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=str(path))
        if is_main:
            (path / "trainer_state.json").write_text(
                json.dumps({"step": step, "epoch": epoch, "scheduler": scheduler.state_dict()})
            )
            (resume_root / "latest").write_text(path.name)
            for old in sorted(resume_root.glob("step_*")):
                if old.name != path.name:
                    shutil.rmtree(old, ignore_errors=True)
            log(f"saved resume state {path}")
        barrier()

    def load_resume_state():
        latest = resume_root / "latest"
        if not latest.exists():
            return 0, 0
        path = resume_root / latest.read_text().strip()
        model_sd, optim_sd = get_state_dict(model, optimizer)
        state = {"model": model_sd, "optim": optim_sd}
        dcp.load(state, checkpoint_id=str(path), planner=DefaultLoadPlanner(allow_partial_load=True))
        set_state_dict(model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optim"])
        trainer_state = json.loads((path / "trainer_state.json").read_text())
        scheduler.load_state_dict(trainer_state["scheduler"])
        log(f"resumed from {path} (step {trainer_state['step']})")
        return int(trainer_state["step"]), int(trainer_state["epoch"])

    # ── training loop ─────────────────────────────────────────────────
    max_grad_norm = float(tcfg.get("max_grad_norm", 1.0))
    log_every = int(tcfg.get("logging_frequency", 10))
    eval_every = int(tcfg.get("eval_interval", 1000))
    save_every = int(tcfg.get("save_interval", 5000))

    time_limit_s = float(tcfg.get("time_limit_hours", 0) or 0) * 3600
    step, epoch = load_resume_state() if tcfg.get("resume", True) else (0, 0)

    model.train()
    if tcfg.get("eval_at_start", True) and step == 0:
        evaluate(0)

    train_sampler.set_epoch(epoch)
    data_iter = iter(train_loader)
    window = {"loss": 0.0, "n": 0, "data_time": 0.0, "step_time": 0.0, "samples": 0, "extra": {}}
    t_last = time.time()
    while step < max_steps:
        t_step = time.time()
        loss_sum = 0.0
        for micro in range(accum):
            t_data = time.time()
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                train_sampler.set_epoch(epoch)
                data_iter = iter(train_loader)
                batch = next(data_iter)
            window["data_time"] += time.time() - t_data
            if distributed:
                model.set_requires_gradient_sync(micro == accum - 1)
            with autocast():
                out = model(batch)
            loss = out["action_loss"]
            if "gate_loss" in out:
                # eye_gate.gated_grad=gate: the gated-path loss trains the gate only. Its graph is disjoint from
                # action_loss's (detached VLM features), so backpropagate it first and drop what it left on the DiT.
                if accum != 1:
                    raise ValueError("eye_gate.gated_grad=gate needs gradient_accumulation_steps 1")
                saved = [p.grad.clone() if p.grad is not None else None for p in dit_params]
                out["gate_loss"].backward()
                for p, g in zip(dit_params, saved):
                    p.grad = g
            (loss / accum).backward()
            loss_sum += loss.detach().float().item()
            for key, value in out.items():  # extra scalars (e.g. gate losses / keep ratios)
                if key != "action_loss" and torch.is_tensor(value) and value.numel() == 1:
                    window["extra"][key] = window["extra"].get(key, 0.0) + value.float().item() / accum
            window["samples"] += len(batch)
        if getattr(model, "diagnostics", False):  # gradient norm per top-level module (before clipping)
            sq = {}
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                g = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
                key = "vision" if ".visual." in name else name.split(".")[0]
                sq[key] = sq.get(key, 0.0) + g.float().pow(2).sum()
            keys = sorted(sq)
            vec = torch.stack([torch.as_tensor(sq[k], device=device, dtype=torch.float32) for k in keys])
            if distributed:
                dist.all_reduce(vec)
            for k, v in zip(keys, vec.sqrt().tolist()):
                window["extra"][f"gradnorm_{k}"] = window["extra"].get(f"gradnorm_{k}", 0.0) + v
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        window["loss"] += loss_sum / accum
        window["n"] += 1
        window["step_time"] += time.time() - t_step

        if step % log_every == 0 or step == max_steps:
            stats = torch.tensor([window["loss"], window["n"], window["samples"]], dtype=torch.float64, device=device)
            if distributed:
                dist.all_reduce(stats)
            elapsed = time.time() - t_last
            mean_loss = (stats[0] / stats[1]).item()
            payload = {
                "step": step,
                "train/action_loss": mean_loss,
                "train/grad_norm": float(grad_norm),
                "train/epoch": epoch,
                "train/samples_per_s": stats[2].item() / elapsed,
                "train/step_time_s": window["step_time"] / window["n"],
                "train/data_wait_frac": window["data_time"] / max(window["step_time"], 1e-6),
            }
            if window["extra"]:
                keys = sorted(window["extra"])
                extra = torch.tensor([window["extra"][k] for k in keys], dtype=torch.float64, device=device)
                if distributed:
                    dist.all_reduce(extra)
                for k, v in zip(keys, (extra / stats[1]).tolist()):
                    payload[f"train/{k}"] = v
            for g in optimizer.param_groups:
                payload[f"train/lr_{g['name']}"] = g["lr"]
            log(f"step {step}/{max_steps} loss {mean_loss:.4f} gnorm {float(grad_norm):.3f} "
                f"{payload['train/samples_per_s']:.1f} samples/s data-wait {payload['train/data_wait_frac']:.0%}")
            if wandb is not None:
                wandb.log(payload)
            window = {"loss": 0.0, "n": 0, "data_time": 0.0, "step_time": 0.0, "samples": 0, "extra": {}}
            t_last = time.time()

        if step % eval_every == 0 or step == max_steps:
            evaluate(step)
        if step % save_every == 0 or step == max_steps:
            save_checkpoint(step)
            save_resume_state(step, epoch)
        elif time_limit_s:
            out_of_time = torch.tensor([float(time.time() - t0 > time_limit_s)], device=device)
            if distributed:
                dist.all_reduce(out_of_time, op=dist.ReduceOp.MAX)
            if out_of_time.item() > 0:
                log(f"time limit reached at step {step}; saving resume state and exiting")
                save_resume_state(step, epoch)
                break

    if wandb is not None:
        wandb.finish()
    barrier()
    if distributed:
        dist.destroy_process_group()


def load_weights(model, path, log=print):
    """Load a checkpoint saved by this script (directory or model.safetensors) into an unsharded model."""
    from safetensors.torch import load_file

    path = Path(path)
    state = load_file(str(path / "model.safetensors" if path.is_dir() else path))
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith("lm_head.weight")]
    log(f"loaded {path}: missing {len(missing)} unexpected {len(unexpected)}")
    if hasattr(model.qwen_vl_interface.model, "tie_weights"):
        model.qwen_vl_interface.model.tie_weights()


if __name__ == "__main__":
    sys.exit(main())
