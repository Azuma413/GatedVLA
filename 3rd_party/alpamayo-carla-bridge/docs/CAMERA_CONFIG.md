# Camera Configuration Guide

How the four-camera rig is set up in CARLA and how to adjust it for your needs.

---

## Default Camera Rig

All cameras are **RGB sensors** (`sensor.camera.rgb`) attached to the ego vehicle.

| Camera | FOV (°) | Position (x, y, z) | Yaw (°) | Purpose |
|--------|---------|---------------------|---------|---------|
| `front_wide` | 120 | 1.5, 0.0, 1.6 | 0 | Main forward view, wide peripheral |
| `front_left` | 70 | 1.5, −0.3, 1.6 | −30 | Left overlap zone |
| `front_right` | 70 | 1.5, 0.3, 1.6 | +30 | Right overlap zone |
| `front_tele` | 30 | 1.5, 0.0, 1.6 | 0 | Narrow telephoto — far-ahead detail |

Default resolution: **800 × 600 px** (configurable in `config.yaml`).

---

## Coordinate System

CARLA uses a **left-handed** coordinate system for sensor attachment:

```
        ▲ Z (up)
        │
        │    ▲ X (forward)
        │   ╱
        │  ╱
        │ ╱
        └──────────▶ Y (right)
```

- **x** — forward from the vehicle's center of mass
- **y** — right from the vehicle's center of mass (negative = left)
- **z** — up from the ground plane
- **yaw** — rotation around Z; positive = clockwise when viewed from above (turn right)
- **pitch** — rotation around Y; positive = nose down

### What the default values mean

| Param | Value | Meaning |
|-------|-------|---------|
| `x: 1.5` | 1.5 m ahead of CoM | Roughly at the front bumper / hood |
| `y: 0.0` | Centered | On the vehicle centerline |
| `y: ±0.3` | 30 cm offset | Slight left / right parallax |
| `z: 1.6` | 1.6 m above ground | Approximate windshield height for a sedan |
| `pitch: 0` | Level horizon | Camera looks straight ahead |
| `yaw: ±30` | 30° left / right | Angled side cameras |

---

## Field of View (FOV)

FOV controls how much of the scene is visible horizontally.

```
          FOV = 30° (tele)         FOV = 120° (wide)
            ╱▔╲                      ╱▔▔▔▔▔▔╲
           ╱    ╲                   ╱          ╲
          ╱  ··  ╲                 ╱   ·····    ╲
         ╱  ····  ╲               ╱  ·········   ╲
        ╱  ······  ╲             ╱  ···········   ╲
```

- **Narrow FOV (30°)** — telephoto, high detail on distant objects, small coverage
- **Standard FOV (70°)** — balanced, similar to a human eye's central vision
- **Wide FOV (120°)** — fisheye-like, large coverage, barrel distortion at edges

### Why four cameras?

Alpamayo expects multi-view input mimicking a real autonomous vehicle's camera array.
The four views provide:

1. **Peripheral awareness** (wide + left + right) — ~180° combined coverage
2. **Long-range detail** (tele) — traffic lights, signs, distant vehicles

---

## Adjusting for Different Vehicles

CARLA vehicles have different sizes. The default `x=1.5, z=1.6` works for a sedan
(Tesla Model 3). For other blueprints:

### Finding the right position

```python
import carla

client = carla.Client("localhost", 2000)
world = client.get_world()
bp = world.get_blueprint_library().find("vehicle.tesla.model3")
vehicle = world.spawn_actor(bp, world.get_map().get_spawn_points()[0])

bb = vehicle.bounding_box
print(f"Extent: x={bb.extent.x:.2f}, y={bb.extent.y:.2f}, z={bb.extent.z:.2f}")
print(f"Location: x={bb.location.x:.2f}, y={bb.location.y:.2f}, z={bb.location.z:.2f}")
```

### Guidelines

| Vehicle type | Suggested x | Suggested z | Notes |
|-------------|------------|------------|-------|
| Sedan (Model 3) | 1.5 | 1.6 | Default |
| SUV | 1.6 | 1.9 | Higher roof |
| Truck / Bus | 2.0 | 2.5 | Much taller, longer hood |
| Compact | 1.2 | 1.4 | Shorter, lower |

### Updating config

Edit `config.yaml`:

```yaml
cameras:
  resolution: [800, 600]
  front_wide:
    fov: 120
    x: 2.0     # moved forward for a truck
    y: 0.0
    z: 2.5     # higher up
    pitch: -5   # tilt slightly down
    yaw: 0
  # ... repeat for other cameras
```

---

## Using `setup_carla_cameras.py`

The helper script lets you preview camera placements without running the full loop:

```bash
python scripts/setup_carla_cameras.py              # default config
python scripts/setup_carla_cameras.py --config my.yaml
```

It will:
1. Connect to CARLA
2. Spawn the vehicle
3. Attach all cameras
4. Save one snapshot per camera to `camera_preview/`
5. Clean up and exit

Use this to iterate on positions before a full recording run.

---

## Tips

- **Overlap is good.** Adjacent cameras should share some field of view for depth cues.
- **Avoid looking at the car.** If `x` is too small or `z` too low, you'll see the vehicle's own body.
- **Pitch down slightly** (e.g., `pitch: -5`) for better road coverage at the expense of sky.
- **Higher resolution** gives the model more detail but costs more inference time.
