"""QwenDrivePI traj / unicycle runs (v2, AnchorVLA v2): state, prompt, controls decoding and run verification."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf
from py123d.api import SceneAPI
from py123d.geometry import PoseSE2

from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.qwendrive_pi_config import QwenDrivePIConfig
from py123d_garage.policy.qwendrive_pi.qwendrive_pi_policy import (
    QwenDrivePIPolicy,
    _anchor_problems,  # pyright: ignore[reportPrivateUsage]
    _drive_prompt,  # pyright: ignore[reportPrivateUsage]
    _rollout_unicycle,  # pyright: ignore[reportPrivateUsage]
    _straight_line_target_point,  # pyright: ignore[reportPrivateUsage]
    _traj_state,  # pyright: ignore[reportPrivateUsage]
    _unstandardize,  # pyright: ignore[reportPrivateUsage]
)

_TRAJ_NAMES = ["x", "y", "heading_rel", "speed", "acc_lon"]
_MEAN, _STD = [-0.0883, -5.48e-05], [0.9012, 0.02454]


def _ego_state(x: float, y: float, yaw: float, speed: float, acc_x: float) -> SimpleNamespace:
    return SimpleNamespace(
        rear_axle_se2=SimpleNamespace(x=x, y=y, yaw=yaw),
        dynamic_state_se3=SimpleNamespace(
            velocity_2d=SimpleNamespace(array=np.array([speed, 0.0])),
            acceleration_3d=SimpleNamespace(array=np.array([acc_x, 0.3, -0.1])),
        ),
    )


class _EgoScene:
    """SceneAPI stand-in serving ego states by iteration; missing iterations return None."""

    def __init__(self, states: dict[int, SimpleNamespace]) -> None:
        self._states = states

    def get_ego_state_se3_at_iteration(self, iteration: int) -> SimpleNamespace | None:
        return self._states.get(iteration)


def _norm_stats() -> dict[str, Any]:
    return {
        "state": {"q01": [-12.0, -0.1, -0.1, 0.0, -2.4], "q99": [0.0, 0.1, 0.1, 35.6, 2.0], "names": _TRAJ_NAMES},
        "action": {
            "q01": [-2.4, -0.09],
            "q99": [1.9, 0.09],
            "names": ["accel", "curvature"],
            "norm": "mean_std",
            "mean": _MEAN,
            "std": _STD,
        },
        "sampler": {"state_horizon": 5, "action_horizon": 25, "dt": 0.1, "image_offsets": [0.1, 0.0]},
    }


def _run_config(anchor_path: str | None = None) -> DictConfig:
    return cast(
        DictConfig,
        OmegaConf.create(
            {
                "framework": {
                    "name": "QwenDrivePI",
                    "anchor": {"enabled": anchor_path is not None, "path": anchor_path},
                    "action_model": {"action_horizon": 25, "state_horizon": 5},
                },
                "datasets": {
                    "vla_data": {
                        "target_distance": 25.0,
                        "state_mode": "traj",
                        "action_mode": "unicycle",
                        "state_target_point": False,
                        "target_point_in_prompt": True,
                        "speed_in_prompt": True,
                    },
                },
            },
        ),
    )


def _policy() -> QwenDrivePIPolicy:
    return QwenDrivePIPolicy(PolicyConfig(qwendrive_pi_config=QwenDrivePIConfig()))


def _write_anchors(path: Path, horizon: int = 25, space: str = "control", std: list[float] | None = None) -> None:
    np.savez(
        path,
        anchors=np.zeros((100, horizon, 2), np.float32),
        residual_std=np.ones((horizon, 2), np.float32),
        space=np.array(space),
        dt=np.array(0.1),
        action_mean=np.array(_MEAN, np.float32),
        action_std=np.array(std or _STD, np.float32),
    )


def test_traj_state_is_oldest_first_in_the_anchor_frame() -> None:
    # The ego drives along +y in the world at 2 m/s, heading pi/2, turning left by 0.01 rad per step.
    states = {
        -step: _ego_state(0.0, 10.0 - 0.2 * step, math.pi / 2 - 0.01 * step, 2.0 - 0.1 * step, 0.5) for step in range(5)
    }
    state = _traj_state(cast(SceneAPI, _EgoScene(states)), state_horizon=5, interval_s=0.1)
    assert state.shape == (5, 5)
    np.testing.assert_allclose(state[-1], [0.0, 0.0, 0.0, 2.0, 0.5], atol=1e-6)
    # Past positions lie behind the anchor: x negative, y ~ 0 in the anchor's frame (x forward, y left).
    np.testing.assert_allclose(state[:, 0], [-0.8, -0.6, -0.4, -0.2, 0.0], atol=1e-5)
    np.testing.assert_allclose(state[:, 1], 0.0, atol=1e-5)
    # Earlier yaws were smaller (the ego turned left since), so heading_rel is negative, CCW positive.
    np.testing.assert_allclose(state[:, 2], [-0.04, -0.03, -0.02, -0.01, 0.0], atol=1e-6)
    np.testing.assert_allclose(state[:, 3], [1.6, 1.7, 1.8, 1.9, 2.0], atol=1e-6)
    np.testing.assert_allclose(state[:, 4], 0.5)


def test_traj_state_extends_a_young_rollout_at_constant_velocity_and_wraps_the_heading() -> None:
    # Heading ~ pi (driving towards -x in the world) at 3 m/s; only iterations 0 and -1 are served.
    states = {0: _ego_state(5.0, 0.0, math.pi - 0.05, 3.0, 0.2), -1: _ego_state(5.3, 0.0, -math.pi + 0.05, 3.0, 0.1)}
    state = _traj_state(cast(SceneAPI, _EgoScene(states)), state_horizon=5, interval_s=0.1)
    # Iterations -4 .. -2 continue the oldest served state (-1) backwards by 0.3 m per step along its heading.
    steps = np.diff(state[:4, :2], axis=0)
    np.testing.assert_allclose(np.linalg.norm(steps, axis=1), 0.3, atol=1e-5)
    np.testing.assert_allclose(state[:4, 2:], np.repeat(state[3:4, 2:], 4, axis=0))
    np.testing.assert_allclose(state[3, 2], 0.1, atol=1e-5)
    assert state[0, 0] < state[1, 0] < state[2, 0] < state[3, 0] < 0.0


def test_rollout_matches_starvla_while_the_speed_stays_positive() -> None:
    from starVLA.model.modules.action_space.unicycle import UnicycleCodec  # pyright: ignore[reportMissingImports]

    rng = np.random.default_rng(0)
    controls = np.stack([rng.normal(0.0, 0.5, (4, 25)), rng.normal(0.0, 0.02, (4, 25))], axis=-1).astype(np.float32)
    v0 = np.array([8.0, 15.0, 4.0, 25.0])
    expected = UnicycleCodec(25, 0.1).decode(controls, v0).numpy()
    np.testing.assert_allclose(_rollout_unicycle(controls, v0, 0.1), expected, atol=1e-4)


def test_rollout_stops_instead_of_rolling_backwards() -> None:
    # Braking at 4 m/s² from 2 m/s stops after 0.5 s; the remaining controls would reverse the ego.
    controls = np.zeros((1, 25, 2), np.float32)
    controls[0, :, 0] = -4.0
    controls[0, :, 1] = 0.05
    xy = _rollout_unicycle(controls, np.array([2.0]), 0.1)[0]
    assert np.all(np.diff(xy[:, 0]) >= -1e-9)
    np.testing.assert_allclose(xy[5:], np.repeat(xy[5:6], 20, axis=0), atol=1e-9)
    np.testing.assert_allclose(xy[4, 0], 0.5, atol=0.02)


def test_controls_decode_from_the_current_speed_without_t0() -> None:
    # Zero normalized controls are the mean controls; with zero mean the ego keeps its speed on a straight line.
    controls = _unstandardize(np.zeros((1, 25, 2), np.float32), np.zeros(2, np.float32), np.ones(2, np.float32))
    xy = _rollout_unicycle(controls, np.array([8.0]), 0.1)
    assert xy.shape == (1, 25, 2)
    np.testing.assert_allclose(xy[0, :, 0], 0.8 * np.arange(1, 26), atol=1e-5)
    np.testing.assert_allclose(xy[0, :, 1], 0.0, atol=1e-9)
    np.testing.assert_allclose(
        _unstandardize(np.ones((1, 2), np.float32), np.array(_MEAN, np.float32), np.array(_STD, np.float32)),
        [[_MEAN[0] + _STD[0], _MEAN[1] + _STD[1]]],
        rtol=1e-6,
    )


def test_prompt_matches_the_training_loader() -> None:
    policy = _policy()
    policy._target_point_in_prompt = True  # pyright: ignore[reportPrivateUsage]
    policy._speed_in_prompt = True  # pyright: ignore[reportPrivateUsage]
    inputs = policy._prompt_inputs(np.array([24.63, -1.25], np.float32), 8.26)  # pyright: ignore[reportPrivateUsage]
    assert inputs == {"v0": 8.26, "lang": "predict trajectory. speed: 8.3 m/s. target point: x 24.6 m, y -1.2 m"}


def test_prompt_copy_matches_starvla() -> None:
    import importlib.util

    import starVLA  # pyright: ignore[reportMissingImports]

    # Loaded from its file: the starVLA.dataloader package pulls training-only dependencies.
    source = Path(cast(str, starVLA.__file__)).parent / "dataloader" / "pai_av_drive.py"
    spec = importlib.util.spec_from_file_location("pai_av_drive_for_test", source)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for target, speed in [
        ((24.63, -1.25), 8.26),
        ((-0.04, 0.0), 0.0),
        ((3.0, 18.96), 31.25),
        (None, 4.0),
        ((1, 2), None),
    ]:
        target_xy = None if target is None else np.array(target, np.float32)
        assert _drive_prompt("predict trajectory", target_xy, speed) == module.drive_prompt(
            "predict trajectory", target_xy, speed
        )


def test_waypoint_runs_get_no_prompt() -> None:
    assert _policy()._prompt_inputs(np.zeros(2, np.float32), 3.0) == {"v0": 3.0}  # pyright: ignore[reportPrivateUsage]


def test_verify_accepts_the_v2_run(tmp_path: Path) -> None:
    _policy()._verify_run(_run_config(), _norm_stats(), tmp_path)  # pyright: ignore[reportPrivateUsage]


def test_verify_accepts_an_anchor_run_with_its_codebook(tmp_path: Path) -> None:
    _write_anchors(tmp_path / "anchors.npz")
    _policy()._verify_run(_run_config("/lus/meta/anchors.npz"), _norm_stats(), tmp_path)  # pyright: ignore[reportPrivateUsage]


def test_verify_refuses_an_anchor_run_without_its_codebook(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="anchor codebook"):
        _policy()._verify_run(_run_config("/lus/meta/anchors.npz"), _norm_stats(), tmp_path)  # pyright: ignore[reportPrivateUsage]


def test_anchor_problems_catch_a_mismatched_codebook(tmp_path: Path) -> None:
    _write_anchors(tmp_path / "short.npz", horizon=20)
    assert any(
        "steps" in problem for problem in _anchor_problems(tmp_path / "short.npz", _norm_stats(), 25, "unicycle")
    )
    _write_anchors(tmp_path / "waypoint.npz", space="waypoint")
    assert any(
        "space" in problem for problem in _anchor_problems(tmp_path / "waypoint.npz", _norm_stats(), 25, "unicycle")
    )
    _write_anchors(tmp_path / "stats.npz", std=[1.0, 1.0])
    assert any(
        "action_std" in problem for problem in _anchor_problems(tmp_path / "stats.npz", _norm_stats(), 25, "unicycle")
    )


def test_verify_refuses_a_q01_q99_unicycle_run(tmp_path: Path) -> None:
    norm_stats = _norm_stats()
    del norm_stats["action"]["norm"]
    with pytest.raises(ValueError, match="q01 / q99 unicycle"):
        _policy()._verify_run(_run_config(), norm_stats, tmp_path)  # pyright: ignore[reportPrivateUsage]


def test_rollout_holds_a_fitted_stop_at_rest() -> None:
    from starVLA.model.modules.action_space.unicycle import UnicycleCodec  # pyright: ignore[reportMissingImports]

    # A logged stop from 5 m/s within 1 s; the training fit's controls dip below zero speed and come back.
    t = np.arange(1, 26) * 0.1
    x = np.where(t < 1.0, 5.0 * t - 2.5 * t**2, 2.5)
    codec = UnicycleCodec(25, 0.1)
    controls = codec.encode(np.stack([x, np.zeros_like(x)], axis=-1), np.zeros(25), 5.0)[None].astype(np.float32)
    xy = _rollout_unicycle(controls, np.array([5.0]), 0.1)[0]
    expected = codec.decode(controls, np.array([5.0])).numpy()[0]
    np.testing.assert_allclose(xy[-1], expected[-1], atol=0.01)
    np.testing.assert_allclose(xy[15:, 0], xy[-1, 0], atol=0.01)


def test_straight_line_target_point_follows_training_on_a_curve() -> None:
    # A left quarter circle of radius 10 m after 5 m straight, sampled every 0.1 m; the ego sits at its start.
    straight = np.stack([np.arange(0.0, 5.0, 0.1), np.zeros(50)], axis=1)
    angles = np.arange(0.0, math.pi / 2, 0.01)
    arc = np.stack([5.0 + 10.0 * np.sin(angles), 10.0 - 10.0 * np.cos(angles)], axis=1)
    after = np.stack([np.full(300, 15.0), 10.0 + np.arange(0.0, 30.0, 0.1)], axis=1)
    xy = np.concatenate([straight, arc, after])
    arc_m = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    scene = SimpleNamespace(
        get_ego_state_se3_at_iteration=lambda iteration: SimpleNamespace(rear_axle_se2=PoseSE2(x=0.0, y=0.0, yaw=0.0)),
        get_route=lambda: (None, arc_m, np.concatenate([xy, np.zeros((len(xy), 1))], axis=1)),
        get_route_progress_at_iteration=lambda iteration: 0.0,
    )
    target = _straight_line_target_point(cast(SceneAPI, scene), 25.0)
    # The first route point 25 m away in a straight line, beyond the point 25 m along the arc.
    assert abs(float(np.linalg.norm(target)) - 25.0) < 0.15
    arc_point = xy[np.searchsorted(arc_m, 25.0)]
    assert target[1] > arc_point[1] + 3.0
