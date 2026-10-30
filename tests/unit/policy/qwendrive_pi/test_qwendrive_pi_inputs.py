"""QwenDrivePI inputs: training image sizes, normalization, ego history, closed-loop protocol."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from py123d.api import SceneAPI

from py123d_garage.config.schema.policy.qwendrive_pi_config import QwenDrivePIConfig
from py123d_garage.policy.qwendrive_pi.qwendrive_pi_policy import (
    _AveragedFlowSampler,  # pyright: ignore[reportPrivateUsage]
    _ego_dynamics,  # pyright: ignore[reportPrivateUsage]
    _GraphedFlowSampler,  # pyright: ignore[reportPrivateUsage]
    _normalize,  # pyright: ignore[reportPrivateUsage]
    _training_size,  # pyright: ignore[reportPrivateUsage]
    _unnormalize,  # pyright: ignore[reportPrivateUsage]
)


def _ego_state(speed: float, yaw: float) -> SimpleNamespace:
    return SimpleNamespace(
        rear_axle_se2=SimpleNamespace(yaw=yaw),
        dynamic_state_se3=SimpleNamespace(
            velocity_2d=SimpleNamespace(array=np.array([speed, 0.0])),
            acceleration_3d=SimpleNamespace(array=np.array([0.5, -0.2, 0.1])),
            angular_velocity=SimpleNamespace(array=np.array([0.0, 0.0, 0.3])),
        ),
    )


class _EgoScene:
    """SceneAPI stand-in serving ego states by iteration; missing iterations return None."""

    def __init__(self, states: dict[int, SimpleNamespace]) -> None:
        self._states = states

    def get_ego_state_se3_at_iteration(self, iteration: int) -> SimpleNamespace | None:
        return self._states.get(iteration)


def test_training_sizes_match_the_pai_av_loader() -> None:
    config = QwenDrivePIConfig()
    assert _training_size(config, 921_600, 32) == (1280, 704)
    assert _training_size(config, 174_080, 32) == (544, 288)


def test_normalization_round_trips() -> None:
    q01, q99 = np.array([0.0, -2.0], np.float32), np.array([10.0, 2.0], np.float32)
    values = np.array([[5.0, 0.0], [10.0, -2.0]], np.float32)
    normalized = _normalize(values, q01, q99)
    np.testing.assert_allclose(normalized, [[0.0, 0.0], [1.0, -1.0]], atol=1e-6)
    np.testing.assert_allclose(_unnormalize(normalized, q01, q99), values, atol=1e-5)


def test_ego_dynamics_are_oldest_first_and_pad_a_young_rollout() -> None:
    scene = _EgoScene({0: _ego_state(8.0, math.pi - 0.05), -1: _ego_state(7.0, -math.pi + 0.05)})
    dynamics = _ego_dynamics(cast(SceneAPI, scene), state_horizon=5)
    assert dynamics.shape == (5, 8)
    # Iterations -4 .. -2 are not served yet and repeat the oldest served state (-1).
    np.testing.assert_allclose(dynamics[:4, 0], 7.0)
    assert dynamics[-1, 0] == 8.0
    np.testing.assert_allclose(dynamics[-1, 1:7], [0.5, -0.2, 0.1, 0.0, 0.0, 0.3], atol=1e-6)
    # The heading relative to the anchor wraps across +-pi.
    np.testing.assert_allclose(dynamics[:4, 7], 0.1, atol=1e-5)
    assert dynamics[-1, 7] == 0.0


def test_config_mirrors_the_h25_run_and_replans_every_half_second() -> None:
    config = QwenDrivePIConfig()
    assert config.trajectory_num_steps == 25
    assert config.required_history_duration_us == 400_000
    assert config.required_past_camera_interval_us == 100_000
    assert config.replan_interval_us == 500_000


def test_averaged_sampler_batches_the_samples_and_averages_per_example() -> None:
    seen: list[tuple[int, ...]] = []

    def sample(vl_embs_list: list[torch.Tensor], state: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        seen.append(
            (len({id(tensor) for tensor in vl_embs_list}), vl_embs_list[0].shape[0], kwargs["query_bias"].shape[0])
        )
        # Sample k of example b: state + k, so the mean over k = 0..3 is state + 1.5.
        return state + torch.arange(4.0).repeat(state.shape[0] // 4)[:, None, None]

    sampler = _AveragedFlowSampler(sample, num_samples=4)
    embs, state = torch.zeros(2, 3, 4), torch.stack([torch.zeros(5, 2), torch.ones(5, 2)])
    out = sampler([embs, embs], state, encoder_attention_mask=torch.ones(2, 3), query_bias=torch.zeros(2, 25, 2))
    torch.testing.assert_close(out, state + 1.5)
    # One batched pass of 2 x 4 samples, the shared VLM layer still shared.
    assert seen == [(1, 8, 8)]


def test_graphed_sampler_runs_eager_off_cuda() -> None:
    calls: list[Any] = []

    def sample(vl_embs_list: list[torch.Tensor], state: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        calls.append(kwargs)
        return state + 1

    sampler = _GraphedFlowSampler(sample)
    out = sampler([torch.zeros(1, 3, 4)], torch.zeros(1, 5, 10), encoder_attention_mask=torch.ones(1, 3))
    assert torch.equal(out, torch.ones(1, 5, 10))
    assert len(calls) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_graphed_sampler_replays_fresh_inputs_with_and_without_query_bias() -> None:
    def sample(
        vl_embs_list: list[torch.Tensor],
        state: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        query_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = state * 2 + vl_embs_list[0].sum() + encoder_attention_mask.sum()
        return out + query_bias.sum() if query_bias is not None else out

    sampler = _GraphedFlowSampler(sample)
    device = torch.device("cuda")
    embs, mask = torch.ones(1, 3, 4, device=device), torch.ones(1, 3, device=device)
    for value in (1.0, 2.0):
        state = torch.full((1, 5, 5), value, device=device)
        bias = torch.full((1, 25, 2), value, device=device)
        torch.testing.assert_close(
            sampler([embs, embs], state, encoder_attention_mask=mask), sample([embs], state, mask)
        )
        torch.testing.assert_close(
            sampler([embs, embs], state, encoder_attention_mask=mask, query_bias=bias),
            sample([embs], state, mask, bias),
        )
    # One graph per input layout, reused across values.
    assert sampler.num_graphs == 2
