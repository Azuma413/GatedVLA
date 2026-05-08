# Alpamayo–CARLA Bridge 🚗🧠

Connect [NVIDIA Alpamayo-R1](https://huggingface.co/nvidia/Alpamayo-R1-10B) — a Vision-Language-Action (VLA) model for autonomous driving — to the [CARLA](https://carla.org/) open-source simulator. Drive a virtual Tesla Model 3 using Alpamayo's predicted trajectories, record every frame, and produce a composite analysis video.

> **CPU-only by default.** This project is designed to run inference on CPU with PyTorch SDPA attention so you don't need a beefy GPU.

---

## Features

| Feature | Description |
|---------|-------------|
| **4-camera rig** | front-wide (120°), front-left (70°), front-right (70°), front-tele (30°) |
| **Alpamayo inference** | 64-waypoint trajectory (6.4 s horizon) + natural-language reasoning trace |
| **PID controller** | Lateral + longitudinal PID converts waypoints → throttle / steer / brake |
| **Per-step recording** | JPEG frames, JSONL telemetry, metadata snapshot |
| **Composite video** | Multi-view + BEV trajectory map + reasoning overlay + status bar |
| **Crash detection** | Collision sensor stops the loop immediately |
| **Synchronous mode** | CARLA waits for inference — deterministic, no frame drops |

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| CARLA simulator | **0.9.15+** (server must be running before you launch the bridge) |
| Python | **3.10 – 3.12** |
| ffmpeg | Any recent version (`brew install ffmpeg` / `apt install ffmpeg`) |
| HuggingFace account | Access to the **gated** model `nvidia/Alpamayo-R1-10B` — request access on the model page, then `huggingface-cli login` |

---

## Installation

### Windows (primary target)

```powershell
# 1. Clone
git clone https://github.com/Jonas-a11y/alpamayo-carla-bridge.git
cd alpamayo-carla-bridge

# 2. Create a venv
python -m venv .venv
.venv\Scripts\activate

# 3. Install the CARLA Python API
#    (copy the .whl from your CARLA installation, e.g. PythonAPI\carla\dist\)
pip install path\to\carla-0.9.15-cpXX-cpXX-win_amd64.whl

# 4. Install remaining deps
pip install -r requirements.txt

# 5. Install ffmpeg (if not installed)
#    Option A: winget install ffmpeg
#    Option B: download from https://ffmpeg.org/download.html and add to PATH

# 6. Log in to HuggingFace (needed for gated model download)
huggingface-cli login
```

### Linux / macOS

```bash
# 1. Clone
git clone https://github.com/Jonas-a11y/alpamayo-carla-bridge.git
cd alpamayo-carla-bridge

# 2. Create a venv
python -m venv .venv && source .venv/bin/activate

# 3. Install the CARLA Python API
pip install /path/to/carla-0.9.15-cpXX-cpXX-linux_x86_64.whl

# 4. Install remaining deps
pip install -r requirements.txt

# 5. Log in to HuggingFace
huggingface-cli login
```

---

## Quick Start

### Windows

```powershell
# Terminal 1 — start CARLA server
CarlaUE4.exe -quality-level=Low

# Terminal 2 — run the bridge
python scripts\run_drive.py                       # uses config.yaml defaults
python scripts\run_drive.py --config my.yaml      # custom config
python scripts\run_drive.py --max-steps 120       # override step limit
```

### Linux / macOS

```bash
# Terminal 1 — start CARLA server
./CarlaUE4.sh -quality-level=Low -RenderOffScreen

# Terminal 2 — run the bridge
python scripts/run_drive.py                       # uses config.yaml defaults
python scripts/run_drive.py --config my.yaml      # custom config
python scripts/run_drive.py --max-steps 120       # override step limit
```

The driving loop will:

1. Connect to CARLA and load the configured map (default: `Town03`).
2. Spawn the ego vehicle at a random point.
3. Attach four cameras + a collision sensor.
4. Load Alpamayo on CPU.
5. Run up to **60 steps** (configurable) or until a collision.
6. Save recordings to `recordings/<session_id>/`.
7. Auto-compose a composite video → `recordings/<session_id>/output.mp4`.

### Compose video from an existing recording

```bash
python scripts/compose_video.py recordings/<session_id>
```

---

## Configuration

All settings live in **`config.yaml`**. Key sections:

| Section | What it controls |
|---------|-----------------|
| `carla` | Server host/port, map name, sync mode, tick delta |
| `vehicle` | Blueprint ID |
| `cameras` | Resolution, per-camera FOV / position / orientation |
| `alpamayo` | Model ID, device (`cpu` / `cuda`), attention impl |
| `controller` | PID gains, target speed, actuator limits |
| `run` | `max_steps` — hard cap on driving loop iterations |
| `recording` | Enable/disable, FPS, output video resolution |

> **Switching to GPU:** set `alpamayo.device: "cuda"` and optionally change `attn_implementation` to `"flash_attention_2"` if your card supports it.

---

## Video Output

The composite video layout:

```
┌─────────────────────────────────────────────┐
│              front_tele (Zoom)               │
├────────────┬─────────────┬──────────────────┤
│ front_left │ front_wide  │   front_right    │
├────────────┴─────────────┴──────────────────┤
│  BEV Trajectory Map     │  Reasoning Trace  │
│  blue = predicted       │  (typewriter fx)  │
│  red  = actual path     │                   │
│  green = ego position   │                   │
├──────────────────────────┴──────────────────┤
│  Status Bar: Speed | Step X/60 | Timer      │
└─────────────────────────────────────────────┘
```

---

## Project Structure

```
alpamayo-carla-bridge/
├── config.yaml                 # All tunables
├── requirements.txt
├── bridge/
│   ├── __init__.py
│   ├── carla_client.py         # CARLA connection & world setup
│   ├── camera_manager.py       # Spawn & manage camera sensors
│   ├── ego_motion.py           # Ego-state history tracker
│   ├── alpamayo_inference.py   # Model loading & inference
│   ├── trajectory_controller.py# PID controller
│   ├── recorder.py             # Per-step frame & telemetry saver
│   ├── video_composer.py       # Composite video renderer
│   └── utils.py                # Shared helpers
├── scripts/
│   ├── run_drive.py            # Main entry point
│   ├── compose_video.py        # Standalone video composer
│   └── setup_carla_cameras.py  # Camera debug / preview utility
├── docs/
│   ├── ARCHITECTURE.md         # Pipeline & data-flow diagrams
│   └── CAMERA_CONFIG.md        # Camera placement guide
└── recordings/                 # Auto-created per session
```

---

## Known Limitations

- **Domain gap** — Alpamayo was trained on real-world driving data; CARLA's rendered world looks different. Expect degraded trajectory quality.
- **CPU speed** — Inference on CPU is *slow* (seconds per step). Synchronous mode ensures correctness but makes real-time driving impossible.
- **No traffic** — The default config does not spawn NPC vehicles or pedestrians. Add them via CARLA's Traffic Manager if desired.
- **Model gating** — `nvidia/Alpamayo-R1-10B` is a gated model. You must request and be granted access on HuggingFace before downloading.
- **CARLA API compatibility** — Tested with CARLA 0.9.15. Older versions may have breaking API changes.

---

## References

- **Alpamayo model:** <https://huggingface.co/nvidia/Alpamayo-R1-10B>
- **Alpamayo code / paper:** <https://github.com/NVlabs/alpamayo>
- **CARLA simulator:** <https://carla.org/>
- **CARLA Python API docs:** <https://carla.readthedocs.io/>

---

## License

MIT — see individual dependencies for their licenses.
