# Architecture

End-to-end data flow for the Alpamayo–CARLA bridge.

---

## High-Level Pipeline

```
┌──────────────┐
│  CARLA Server│
│  (UE4 world) │
└──────┬───────┘
       │  TCP (localhost:2000)
       ▼
┌──────────────────────────────────────────────────────────────┐
│                       Bridge Process                         │
│                                                              │
│  ┌─────────────┐   ┌────────────────┐   ┌────────────────┐  │
│  │ CarlaClient │──▶│ CameraManager  │──▶│  4× RGB frames │  │
│  │ (connect,   │   │ (spawn sensors,│   │  (800×600 px)  │  │
│  │  spawn ego, │   │  collect imgs) │   └───────┬────────┘  │
│  │  sync tick) │   └────────────────┘           │            │
│  └──────┬──────┘                                │            │
│         │                                       ▼            │
│         │         ┌──────────────┐    ┌─────────────────┐    │
│         │         │  EgoMotion   │───▶│ AlpamayoInfer.  │    │
│         │         │  (velocity,  │    │ (VLA model,     │    │
│         │         │   heading,   │    │  CPU / SDPA)    │    │
│         │         │   history)   │    └────────┬────────┘    │
│         │         └──────────────┘             │             │
│         │                                      │             │
│         │              trajectory (64 pts)      │             │
│         │              + reasoning text         │             │
│         │                                      ▼             │
│         │                          ┌──────────────────┐      │
│         │                          │ TrajectoryCtrl   │      │
│         │                          │ (PID lateral +   │      │
│         │                          │  PID longitudinal│      │
│         │                          └────────┬─────────┘      │
│         │                                   │                │
│         │          VehicleControl            │                │
│         │   (throttle, steer, brake)        │                │
│         ◀───────────────────────────────────┘                │
│         │                                                    │
│         │  apply_control() + world.tick()                    │
│         │                                                    │
│  ┌──────▼──────┐                                             │
│  │  Recorder   │  saves: JPEG frames, JSONL telemetry,      │
│  │             │         metadata.json                       │
│  └──────┬──────┘                                             │
│         │  (on loop exit)                                    │
│         ▼                                                    │
│  ┌──────────────┐                                            │
│  │VideoComposer │  reads recording → composite MP4           │
│  └──────────────┘                                            │
└──────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Sequence

```
 CARLA Server          Bridge                    Alpamayo (CPU)
 ────────────          ──────                    ──────────────
      │                   │                            │
      │◀── connect ───────│                            │
      │── world handle ──▶│                            │
      │                   │                            │
      │◀── spawn ego ─────│                            │
      │── actor handle ──▶│                            │
      │                   │                            │
      │◀── spawn cams ────│                            │
      │── sensor actors ─▶│                            │
      │                   │                            │
      │                   │── load model ─────────────▶│
      │                   │◀─ model ready ─────────────│
      │                   │                            │
      │  ┌─── LOOP (max 60 steps) ────────────────┐   │
      │  │                │                        │   │
      │  │  collect frames│                        │   │
      │  │◀───────────────│                        │   │
      │  │── 4 images ───▶│                        │   │
      │  │                │                        │   │
      │  │                │── inference(imgs, ego)─┼──▶│
      │  │                │◀─ trajectory + text ───┼───│
      │  │                │                        │   │
      │  │                │── PID → control ───────│   │
      │  │                │                        │   │
      │  │                │── recorder.save_step ──│   │
      │  │                │                        │   │
      │  │◀── apply ctrl ─│                        │   │
      │  │◀── world.tick()│                        │   │
      │  │                │                        │   │
      │  └──── step += 1 ─┘  (or crash → break)   │   │
      │                   │                            │
      │                   │── compose_video ───────────│
      │                   │                            │
```

---

## Data Flow: Recording

```
Each step produces:
  • 4 JPEG images  →  recordings/{session}/frames/step_XXXX_{cam}.jpg
  • 1 JSONL line   →  recordings/{session}/data.jsonl

At session start:
  • 1 metadata.json  →  config snapshot, spawn point, world name, timestamps

At session end:
  • VideoComposer reads the recording directory and writes output.mp4
```

---

## Synchronous Mode

CARLA is configured with a **fixed time-step** (`fixed_delta: 0.1` → 10 Hz simulation).
The bridge calls `world.tick()` explicitly after applying each control command.
This guarantees:

1. No simulation frames are skipped while we wait for CPU inference.
2. Physics remain deterministic across runs.
3. Cameras always produce exactly one frame per tick.

The trade-off is wall-clock speed: each step takes as long as Alpamayo inference
(potentially several seconds on CPU).

---

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `carla_client.py` | TCP connection, world loading, ego spawn, sync-mode setup, cleanup |
| `camera_manager.py` | Sensor blueprint creation, attachment transforms, frame queues |
| `ego_motion.py` | Read vehicle transform + velocity each tick, maintain rolling history |
| `alpamayo_inference.py` | HuggingFace model load, image preprocessing, forward pass, output parsing |
| `trajectory_controller.py` | Dual PID (lateral + longitudinal), waypoint-to-control conversion |
| `recorder.py` | Session directory creation, per-step JPEG + JSONL writing, metadata |
| `video_composer.py` | Multi-view layout, BEV plot, reasoning overlay, ffmpeg encoding |
| `utils.py` | Config loading, logging setup, coordinate transforms |
