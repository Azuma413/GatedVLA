from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import math
from pathlib import Path
from typing import Any, cast

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from py123d.api import SceneAPI
from py123d.datatypes import EgoStateSE3
from typing_extensions import override

from py123d_garage.api.abstract_policy import AbstractPolicy, CacheTensorSpec
from py123d_garage.api.abstract_policy_tensors import (
    AbstractFeatures,
    AbstractLabels,
    AbstractPredictions,
    NavigationConditioning,
)
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.qwendrive_pi_config import QwenDrivePIConfig
from py123d_garage.datatypes.trajectory import derive_yaw_from_positions
from py123d_garage.py123d_help.scene_readers import camera_at_anchor, sample_ego_xy, sample_past_camera

LOG = logging.getLogger(__name__)

_EVALUATION_ONLY = "QwenDrivePIPolicy runs a starVLA QwenDrivePI checkpoint for evaluation only; train it in starVLA."

# Per-step ego dynamics the starVLA "drivemoe" state carries before the target point (pai_av_drive.STATE_NAMES).
_DYNAMICS_NAMES = ["speed", "acc_x", "acc_y", "acc_z", "angvel_x", "angvel_y", "angvel_z", "heading_rel"]
_STATE_NAMES = [*_DYNAMICS_NAMES, "target_x", "target_y"]


@dataclasses.dataclass(frozen=True)
class QwenDrivePIFeatures(AbstractFeatures):
    # Context frames before the current one, oldest first, at the training history size.
    history_frames: jt.UInt8[torch.Tensor, "*batch history height width 3"]
    # The current frame at the training size.
    current_frame: jt.UInt8[torch.Tensor, "*batch current_height current_width 3"]
    # Un-normalized per-step ego dynamics, oldest first, the anchor last (see _DYNAMICS_NAMES).
    ego_dynamics: jt.Float[torch.Tensor, "*batch state_horizon 8"]


@dataclasses.dataclass(frozen=True)
class QwenDrivePILabels(AbstractLabels):
    # The logged future positions on the policy's trajectory grid, relative to the anchor.
    trajectory_xy: jt.Float[torch.Tensor, "*batch poses 2"]


@dataclasses.dataclass(frozen=True)
class QwenDrivePIPredictions(AbstractPredictions):
    waypoints_xy: jt.Float[torch.Tensor, "*batch poses 2"]

    @property
    @override
    def ego_trajectory_xy(self) -> jt.Float[torch.Tensor, "*batch poses 2"]:
        """Inherited, see superclass."""
        return self.waypoints_xy

    @property
    @override
    def ego_trajectory_se2(self) -> jt.Float[torch.Tensor, "*batch poses 3"]:
        """Inherited, see superclass."""
        yaw = derive_yaw_from_positions(self.waypoints_xy)
        return torch.cat([self.waypoints_xy, yaw.unsqueeze(-1)], dim=-1)


class QwenDrivePIPolicy(AbstractPolicy[QwenDrivePIFeatures, QwenDrivePILabels, QwenDrivePIPredictions]):
    # The starVLA Qwen_Drive_PI framework; starVLA is untyped.
    model: Any

    def __init__(self, config: PolicyConfig):
        """Initializes the QwenDrivePI policy from its config block."""
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        assert config.qwendrive_pi_config is not None, (
            "the target names QwenDrivePIPolicy but no qwendrive_pi_config block is declared"
        )
        self._config: QwenDrivePIConfig = config.qwendrive_pi_config
        self._checkpoint_file: Path | None = None
        self._state_q01 = np.zeros(len(_STATE_NAMES), dtype=np.float32)
        self._state_q99 = np.ones(len(_STATE_NAMES), dtype=np.float32)
        self._action_q01 = np.zeros(2, dtype=np.float32)
        self._action_q99 = np.ones(2, dtype=np.float32)
        # (width, height) of the current and the history frames, read from the run's pixel budgets.
        self._current_size = (0, 0)
        self._history_size = (0, 0)

    @property
    @override
    def policy_config(self) -> QwenDrivePIConfig:
        return self._config

    @override
    def initialize(self, evaluation_checkpoint_file: str | None = None) -> None:
        """
        Routes the starVLA checkpoint to build_model, which reads the architecture from the
        bundled starVLA run config; the base strict torch.load does not apply.

        Args:
            evaluation_checkpoint_file: model.safetensors of the bundle built by scripts/pretrained/qwendrive_pi.sh.
        """
        if evaluation_checkpoint_file is None:
            raise ValueError(
                "QwenDrivePIPolicy is inference-only; it requires policy_config.evaluation_checkpoint_file.",
            )
        self._checkpoint_file = Path(evaluation_checkpoint_file)
        self.build_model(from_local_checkpoint=True)

    @override
    def build_model(self, from_local_checkpoint: bool) -> None:
        del from_local_checkpoint
        assert self._checkpoint_file is not None, "initialize routes the checkpoint here"
        if self._config.linear_attention_kernels not in ("auto", "fla", "torch"):
            raise ValueError(
                f"linear_attention_kernels must be auto / fla / torch, got {self._config.linear_attention_kernels}",
            )
        bundle_dir = self._checkpoint_file.parent
        starvla_config = cast(DictConfig, OmegaConf.load(bundle_dir / self._config.starvla_config_file))
        norm_stats = json.loads((bundle_dir / self._config.norm_stats_file).read_text())
        self._verify_run(starvla_config, norm_stats)

        state_stats, action_stats = norm_stats["state"], norm_stats["action"]
        self._state_q01 = np.asarray(state_stats["q01"], dtype=np.float32)
        self._state_q99 = np.asarray(state_stats["q99"], dtype=np.float32)
        self._action_q01 = np.asarray(action_stats["q01"], dtype=np.float32)
        self._action_q99 = np.asarray(action_stats["q99"], dtype=np.float32)
        drive = starvla_config.framework.drive
        self._current_size = _training_size(self._config, int(drive.current_image_pixels), int(drive.image_factor))
        self._history_size = _training_size(self._config, int(drive.history_image_pixels), int(drive.image_factor))

        # The checkpoint holds every weight, so the VLM is built from its config alone.
        # starVLA picks the VLM interface by the base_vlm name, so the bundled directory keeps the repository name.
        base_vlm_name = Path(str(starvla_config.framework.qwenvl.base_vlm)).name
        starvla_config.framework.qwenvl.base_vlm = str(bundle_dir / self._config.base_vlm_dir / base_vlm_name)
        starvla_config.framework.qwenvl.load_pretrained_weights = False
        starvla_config.trainer.pretrained_checkpoint = None
        if self._config.num_inference_timesteps is not None:
            starvla_config.framework.action_model.num_inference_timesteps = self._config.num_inference_timesteps

        import transformers
        from starVLA.model.framework.VLM4A.QwenDrivePI import (  # pyright: ignore[reportMissingImports]
            Qwen_Drive_PI,  # pyright: ignore[reportUnknownVariableType]
        )

        # starVLA's chat-template call warns on every query under transformers 5; only errors are kept.
        transformers.logging.set_verbosity_error()

        # The checkpoint is bf16 and training ran FSDP with bf16 parameters, so the model is built in bf16.
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = cast("Any", Qwen_Drive_PI(starvla_config))
        finally:
            torch.set_default_dtype(default_dtype)
        if self._config.linear_attention_kernels == "torch":
            _use_torch_linear_attention()
        _load_safetensors_in_place(model, self._checkpoint_file)
        model.eval()
        if self._config.cuda_graph_expert:
            model.action_model.predict_action = _GraphedFlowSampler(model.action_model.predict_action)
        self.model = model

    def _verify_run(self, starvla_config: DictConfig, norm_stats: dict[str, Any]) -> None:
        """Refuses a starVLA run whose input or output layout this policy's config does not mirror."""
        framework = starvla_config.framework
        vla_data = starvla_config.datasets.vla_data
        sampler: dict[str, Any] = norm_stats.get("sampler", {})
        config = self._config
        problems: list[str] = []
        if framework.name != "QwenDrivePI":
            problems.append(f"framework {framework.name} is not QwenDrivePI")
        if framework.get("anchor", {}).get("enabled", False):
            problems.append("decision anchors are not bundled")
        if (
            vla_data.get("state_mode", "drivemoe") != "drivemoe"
            or vla_data.get("action_mode", "waypoints") != "waypoints"
        ):
            problems.append(f"state_mode {vla_data.get('state_mode')} / action_mode {vla_data.get('action_mode')}")
        if vla_data.get("target_point_in_prompt", False) or vla_data.get("speed_in_prompt", False):
            problems.append("prompts with the target point or speed")
        if norm_stats["state"]["names"] != _STATE_NAMES or norm_stats["action"]["names"] != ["x", "y"]:
            problems.append(f"state {norm_stats['state']['names']} / action {norm_stats['action']['names']}")
        if norm_stats["action"].get("norm", "q01_q99") == "mean_std":
            problems.append("mean / std action normalization")

        dt_us = round(float(sampler.get("dt", 0.1)) * 1e6)
        action_horizon = int(framework.action_model.action_horizon)
        image_offsets_us = sorted(round(float(offset) * 1e6) for offset in sampler.get("image_offsets", [0.1, 0.0]))
        expected: dict[str, tuple[Any, Any]] = {
            "trajectory_interval_us": (config.trajectory_interval_us, dt_us),
            "trajectory_horizon_us": (config.trajectory_horizon_us, (action_horizon - 1) * dt_us),
            "state_horizon": (config.state_horizon, int(framework.action_model.state_horizon)),
            "required_target_point_distances_m": (
                config.required_target_point_distances_m,
                [float(sampler.get("target_distance", vla_data.target_distance))],
            ),
            "image offsets": (
                [index * config.context_frame_interval_us for index in range(config.num_history_frames + 1)],
                image_offsets_us,
            ),
        }
        problems += [
            f"{name} {mine} vs. trained {trained}" for name, (mine, trained) in expected.items() if mine != trained
        ]
        if problems:
            raise ValueError(f"the starVLA run does not match qwendrive_pi_config: {'; '.join(problems)}")

    @override
    def warmup(self) -> None:
        """Compiles the linear-attention kernels and captures the expert graph at the served input shapes."""
        (current_width, current_height), (history_width, history_height) = self._current_size, self._history_size
        example: dict[str, Any] = {
            "image": [Image.new("RGB", (history_width, history_height))] * self._config.num_history_frames
            + [Image.new("RGB", (current_width, current_height))],
            "state": np.zeros((self._config.state_horizon, len(_STATE_NAMES)), dtype=np.float32),
        }
        try:
            self.model.predict_action([example])
        except Exception:
            if self._config.linear_attention_kernels != "auto":
                raise
            LOG.exception("the linear-attention Triton kernels failed; falling back to the torch implementation")
            _use_torch_linear_attention()
        for _ in range(2):
            self.model.predict_action([example])

    @override
    def build_features(
        self,
        scene_api: SceneAPI,
        cached: dict[str, torch.Tensor],
    ) -> QwenDrivePIFeatures:
        del cached
        config = self._config
        camera_id = config.input_camera[scene_api.scene_metadata.dataset]
        anchor = camera_at_anchor(scene_api, camera_id)
        past_cameras = sample_past_camera(
            scene_api=scene_api,
            camera_id=camera_id,
            anchor_timestamp_us=anchor.timestamp.time_us,
            horizon_us=config.num_history_frames * config.context_frame_interval_us,
            interval_us=config.context_frame_interval_us,
        )
        # A rollout younger than the context repeats its oldest served frame, the anchor at the latest.
        history_images: list[npt.NDArray[np.uint8]] = []
        previous = anchor.image
        for camera in past_cameras:
            previous = camera.image if camera is not None else previous
            history_images.append(previous)
        history_images.reverse()

        return QwenDrivePIFeatures(
            history_frames=torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
                np.stack([_resized(image, self._history_size) for image in history_images]),
            ),
            current_frame=torch.from_numpy(_resized(anchor.image, self._current_size)),  # pyright: ignore[reportUnknownMemberType]
            ego_dynamics=torch.from_numpy(_ego_dynamics(scene_api, config.state_horizon)),  # pyright: ignore[reportUnknownMemberType]
        )

    @override
    def build_labels(
        self,
        scene_api: SceneAPI,
        cached: dict[str, torch.Tensor],
    ) -> QwenDrivePILabels:
        """The logged future on the waypoint grid; drawn as ground truth, never trained on."""
        del cached
        trajectory = sample_ego_xy(
            scene_api=scene_api,
            num_steps=self._config.trajectory_num_steps,
            interval_us=self._config.trajectory_interval_us,
            relative_to_anchor=True,
        )
        return QwenDrivePILabels(trajectory_xy=torch.tensor(trajectory.position_xy_array, dtype=torch.float32))

    @override
    def _cache_specs(self) -> dict[str, CacheTensorSpec]:
        return {}

    @override
    def forward(
        self,
        features: QwenDrivePIFeatures,
        navigation: NavigationConditioning,
    ) -> QwenDrivePIPredictions:
        """Samples one trajectory per sample from the flow-matching expert."""
        history_frames = cast(
            "npt.NDArray[np.uint8]",
            features.history_frames.cpu().numpy(),  # pyright: ignore[reportUnknownMemberType]
        )
        current_frames = cast(
            "npt.NDArray[np.uint8]",
            features.current_frame.cpu().numpy(),  # pyright: ignore[reportUnknownMemberType]
        )
        dynamics = cast(
            "npt.NDArray[np.float32]",
            features.ego_dynamics.float().cpu().numpy(),  # pyright: ignore[reportUnknownMemberType]
        )
        target_xy = cast(
            "npt.NDArray[np.float32]",
            navigation.target_points[:, 0].float().cpu().numpy(),  # pyright: ignore[reportUnknownMemberType]
        )

        examples: list[dict[str, Any]] = []
        for index in range(current_frames.shape[0]):
            state = np.concatenate(
                [dynamics[index], np.repeat(target_xy[index][None], dynamics.shape[1], axis=0)],
                axis=1,
            )
            examples.append(
                {
                    "image": [Image.fromarray(frame) for frame in history_frames[index]]
                    + [Image.fromarray(current_frames[index])],
                    "state": _normalize(state, self._state_q01, self._state_q99),
                },
            )
        normalized_actions = cast("npt.NDArray[np.float32]", self.model.predict_action(examples)["normalized_actions"])
        # The action starts at t0 itself; the trajectory grid starts one interval later.
        waypoints_xy = _unnormalize(normalized_actions, self._action_q01, self._action_q99)[:, 1:]
        return QwenDrivePIPredictions(
            waypoints_xy=torch.as_tensor(waypoints_xy, dtype=torch.float32, device=features.ego_dynamics.device),
        )

    @override
    def compute_loss(
        self,
        labels: QwenDrivePILabels,
        predictions: QwenDrivePIPredictions,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError(_EVALUATION_ONLY)

    @override
    def loss_weights(self, epoch: int) -> dict[str, float]:
        raise NotImplementedError(_EVALUATION_ONLY)


def _use_torch_linear_attention() -> None:
    """Points Qwen3.5's gated-delta-rule calls back at transformers' torch implementations."""
    from transformers.models.qwen3_5 import modeling_qwen3_5

    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        setattr(modeling_qwen3_5, name, inspect.unwrap(getattr(modeling_qwen3_5, name)))


def _load_safetensors_in_place(model: torch.nn.Module, checkpoint_file: Path) -> None:
    """
    Copies the checkpoint tensor by tensor into the built model, strictly; the tied LM head the
    starVLA trainer leaves out of the file is the only key allowed to be missing.
    """
    from safetensors import safe_open

    targets = model.state_dict()
    with safe_open(str(checkpoint_file), framework="pt", device="cpu") as checkpoint:  # pyright: ignore[reportUnknownVariableType]
        keys = set(cast("list[str]", checkpoint.keys()))  # pyright: ignore[reportUnknownMemberType]
        missing = sorted(key for key in set(targets) - keys if not key.endswith("lm_head.weight"))
        unexpected = sorted(keys - set(targets))
        if missing or unexpected:
            raise RuntimeError(
                f"{checkpoint_file} does not match the QwenDrivePI built from its config: "
                f"missing {missing[:10]} ({len(missing)}), unexpected {unexpected[:10]} ({len(unexpected)}).",
            )
        with torch.no_grad():
            for key in keys:
                targets[key].copy_(checkpoint.get_tensor(key))  # pyright: ignore[reportUnknownMemberType]


class _GraphedFlowSampler:
    """
    The expert's whole denoising loop replayed from one CUDA graph per input shape: the eager loop
    launches thousands of small kernels per plan, so its run time is launch overhead.
    """

    def __init__(self, sample: Any) -> None:
        self._sample = sample
        self._graphs: dict[tuple[Any, ...], tuple[torch.cuda.CUDAGraph, dict[int, torch.Tensor], torch.Tensor]] = {}

    def __call__(
        self,
        vl_embs_list: list[torch.Tensor],
        state: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        query_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            query_bias is not None
            or state is None
            or encoder_attention_mask is None
            or not all(tensor.is_cuda for tensor in [*vl_embs_list, state, encoder_attention_mask])
        ):
            return self._sample(
                vl_embs_list, state, encoder_attention_mask=encoder_attention_mask, query_bias=query_bias
            )
        inputs = [*vl_embs_list, state, encoder_attention_mask]
        # The list repeats one tensor for the blocks that share a VLM layer; the graph keeps that aliasing.
        positions = [next(index for index, other in enumerate(inputs) if other is tensor) for tensor in inputs]
        key = (tuple(positions), *((tensor.shape, tensor.dtype) for tensor in inputs))
        if key not in self._graphs:
            self._graphs[key] = self._capture(inputs, positions)
        graph, static_inputs, static_output = self._graphs[key]
        for position, static in static_inputs.items():
            static.copy_(inputs[position])
        graph.replay()
        return static_output.clone()

    def _capture(
        self,
        inputs: list[torch.Tensor],
        positions: list[int],
    ) -> tuple[torch.cuda.CUDAGraph, dict[int, torch.Tensor], torch.Tensor]:
        static_inputs = {position: inputs[position].clone() for position in sorted(set(positions))}
        static_vl_embs = [static_inputs[position] for position in positions[:-2]]
        state, mask = static_inputs[positions[-2]], static_inputs[positions[-1]]

        def sample() -> torch.Tensor:
            return self._sample(static_vl_embs, state, encoder_attention_mask=mask)

        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())  # pyright: ignore[reportUnknownMemberType]
        with torch.cuda.stream(side_stream):
            for _ in range(2):
                sample()
        torch.cuda.current_stream().wait_stream(side_stream)  # pyright: ignore[reportUnknownMemberType]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = sample()
        return graph, static_inputs, static_output


def _training_size(config: QwenDrivePIConfig, max_pixels: int, factor: int) -> tuple[int, int]:
    """(width, height) the training loader resized the PAI-AV recording to (pai_av_drive.budget_size)."""
    width, height = config.source_image_width, config.source_image_height
    scale = min(1.0, math.sqrt(max_pixels / float(width * height)))
    return (
        max(factor, int(width * scale // factor) * factor),
        max(factor, int(height * scale // factor) * factor),
    )


def _resized(image: npt.NDArray[np.uint8], size: tuple[int, int]) -> npt.NDArray[np.uint8]:
    if (image.shape[1], image.shape[0]) == size:
        return np.ascontiguousarray(image)
    return cast("npt.NDArray[np.uint8]", cv2.resize(image, size, interpolation=cv2.INTER_AREA))


def _ego_dynamics(scene_api: SceneAPI, state_horizon: int) -> npt.NDArray[np.float32]:
    """
    Per-step [speed, acc xyz, angular velocity xyz, heading relative to the anchor] at the anchor and the
    state_horizon - 1 iterations before it, oldest first; a rollout younger than that repeats its oldest state.
    """
    anchor = scene_api.get_ego_state_se3_at_iteration(0)
    assert anchor is not None, "Ego state should be available for feature computation!"
    anchor_yaw = float(anchor.rear_axle_se2.yaw)

    rows: list[npt.NDArray[np.float32]] = []
    previous: EgoStateSE3 = anchor
    for iteration in range(0, -state_horizon, -1):
        ego_state = scene_api.get_ego_state_se3_at_iteration(iteration)
        previous = ego_state if ego_state is not None else previous
        dynamic_state = previous.dynamic_state_se3
        assert dynamic_state is not None, "Ego dynamic state should be available for feature computation!"
        heading_rel = math.remainder(float(previous.rear_axle_se2.yaw) - anchor_yaw, math.tau)
        rows.append(
            np.array(
                [
                    float(np.linalg.norm(dynamic_state.velocity_2d.array)),
                    *dynamic_state.acceleration_3d.array.tolist(),
                    *dynamic_state.angular_velocity.array.tolist(),
                    heading_rel,
                ],
                dtype=np.float32,
            ),
        )
    return np.stack(rows[::-1])


def _normalize(
    values: npt.NDArray[np.float32],
    q01: npt.NDArray[np.float32],
    q99: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """q01 / q99 -> [-1, 1] per dimension, as pai_av_drive.Normalizer."""
    return (2.0 * (values - q01) / np.maximum(q99 - q01, 1e-6) - 1.0).astype(np.float32)


def _unnormalize(
    values: npt.NDArray[np.float32],
    q01: npt.NDArray[np.float32],
    q99: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    return ((np.asarray(values, np.float32) + 1.0) / 2.0 * (q99 - q01) + q01).astype(np.float32)
