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
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image
from py123d.api import SceneAPI
from py123d.datatypes import EgoStateSE3
from py123d.geometry.geometry_index import BoundingBoxSE2Index
from py123d.geometry.transform import abs_to_rel_se2_array
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
from py123d_garage.config.schema.policy.visualization_config import TrajectoryStyleConfig
from py123d_garage.datatypes.trajectory import derive_yaw_from_positions
from py123d_garage.policy.vavam.visualization import render_bev
from py123d_garage.py123d_help.scene_readers import camera_at_anchor, sample_ego_xy, sample_past_camera
from py123d_garage.py123d_help.scene_readers.target_points import InsufficientRouteError

LOG = logging.getLogger(__name__)

_EVALUATION_ONLY = "QwenDrivePIPolicy runs a starVLA QwenDrivePI checkpoint for evaluation only; train it in starVLA."

# Per-step ego dynamics the starVLA "drivemoe" state carries before the target point (pai_av_drive.STATE_NAMES).
_DYNAMICS_NAMES = ["speed", "acc_x", "acc_y", "acc_z", "angvel_x", "angvel_y", "angvel_z", "heading_rel"]
_STATE_NAMES = [*_DYNAMICS_NAMES, "target_x", "target_y"]
# The starVLA "traj" state without the target point (pai_av_drive.TRAJ_STATE_NAMES) and the unicycle controls.
_TRAJ_STATE_NAMES = ["x", "y", "heading_rel", "speed", "acc_lon"]
_UNICYCLE_ACTION_NAMES = ["accel", "curvature"]
_TARGET_STATE_NAMES = ["target_x", "target_y"]
_SPEED_COLUMN = {"drivemoe": _STATE_NAMES.index("speed"), "traj": _TRAJ_STATE_NAMES.index("speed")}

_PAST_TRAJECTORY_STYLE = TrajectoryStyleConfig(color_rgb=[127, 127, 127])
_ANCHOR_REFERENCE_STYLE = TrajectoryStyleConfig(color_rgb=[255, 127, 14], line_thickness_pixel=1, pose_radius_pixel=2)


@dataclasses.dataclass(frozen=True)
class QwenDrivePIFeatures(AbstractFeatures):
    # Context frames before the current one, oldest first, at the training history size.
    history_frames: jt.UInt8[torch.Tensor, "*batch history height width 3"]
    # The current frame at the training size.
    current_frame: jt.UInt8[torch.Tensor, "*batch current_height current_width 3"]
    # Un-normalized per-step ego state, oldest first, the anchor last: _DYNAMICS_NAMES for drivemoe runs (the
    # target point is appended in forward), _TRAJ_STATE_NAMES for traj runs (the target point appended in forward
    # when the run's state carries it).
    ego_state: jt.Float[torch.Tensor, "*batch state_horizon state_dim"]
    # The target point as training placed it (see _straight_line_target_point), in the anchor's rear-axle frame.
    target_point: jt.Float[torch.Tensor, "*batch 2"]


@dataclasses.dataclass(frozen=True)
class QwenDrivePILabels(AbstractLabels):
    # The logged future positions on the policy's trajectory grid, relative to the anchor.
    trajectory_xy: jt.Float[torch.Tensor, "*batch poses 2"]


@dataclasses.dataclass(frozen=True)
class QwenDrivePIPredictions(AbstractPredictions):
    waypoints_xy: jt.Float[torch.Tensor, "*batch poses 2"]
    # AnchorVLA runs: the classifier's soft reference the expert refines, on the same grid; drawn, never served.
    anchor_reference_xy: jt.Float[torch.Tensor, "*batch poses 2"] | None = None

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
        # The run's state_mode ("drivemoe" or "traj"), read from its config when the model is built, and whether
        # its state ends with the target point (always for drivemoe; datasets.vla_data.state_target_point for traj).
        self._state_mode = "drivemoe"
        self._state_target_point = True
        self._state_q01 = np.zeros(len(_STATE_NAMES), dtype=np.float32)
        self._state_q99 = np.ones(len(_STATE_NAMES), dtype=np.float32)
        # Waypoint actions are q01 / q99 normalized, unicycle controls mean / std (starVLA's Normalizer).
        self._action_mean_std = False
        self._action_low = np.zeros(2, dtype=np.float32)
        self._action_high = np.ones(2, dtype=np.float32)
        # traj runs: the starVLA prompt and its switches; unicycle runs: the control step (None for waypoints).
        self._prompt = "predict trajectory"
        self._target_point_in_prompt = False
        self._speed_in_prompt = False
        self._unicycle_dt_s: float | None = None
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
        self._verify_run(starvla_config, norm_stats, bundle_dir)

        vla_data = starvla_config.datasets.vla_data
        self._state_mode = str(vla_data.get("state_mode", "drivemoe"))
        self._state_target_point = self._state_mode == "drivemoe" or bool(vla_data.get("state_target_point", True))
        state_stats, action_stats = norm_stats["state"], norm_stats["action"]
        self._state_q01 = np.asarray(state_stats["q01"], dtype=np.float32)
        self._state_q99 = np.asarray(state_stats["q99"], dtype=np.float32)
        self._action_mean_std = action_stats.get("norm", "q01_q99") == "mean_std"
        low_key, high_key = ("mean", "std") if self._action_mean_std else ("q01", "q99")
        self._action_low = np.asarray(action_stats[low_key], dtype=np.float32)
        self._action_high = np.asarray(action_stats[high_key], dtype=np.float32)
        drive = starvla_config.framework.drive
        self._current_size = _training_size(self._config, int(drive.current_image_pixels), int(drive.image_factor))
        self._history_size = _training_size(self._config, int(drive.history_image_pixels), int(drive.image_factor))
        self._prompt = str(drive.get("prompt", "predict trajectory"))
        self._target_point_in_prompt = bool(vla_data.get("target_point_in_prompt", False))
        self._speed_in_prompt = bool(vla_data.get("speed_in_prompt", False))
        if vla_data.get("action_mode", "waypoints") == "unicycle":
            self._unicycle_dt_s = float(norm_stats.get("sampler", {}).get("dt", 0.1))

        # The checkpoint holds every weight, so the VLM is built from its config alone.
        # starVLA picks the VLM interface by the base_vlm name, so the bundled directory keeps the repository name.
        base_vlm_name = Path(str(starvla_config.framework.qwenvl.base_vlm)).name
        starvla_config.framework.qwenvl.base_vlm = str(bundle_dir / self._config.base_vlm_dir / base_vlm_name)
        starvla_config.framework.qwenvl.load_pretrained_weights = False
        starvla_config.trainer.pretrained_checkpoint = None
        if _anchors_enabled(starvla_config):
            starvla_config.framework.anchor.path = str(bundle_dir / self._config.anchors_file)
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
        _disable_evaluation_extras(model)
        if self._config.num_action_samples > 1:
            model.action_model.predict_action = _AveragedFlowSampler(
                model.action_model.predict_action, self._config.num_action_samples
            )
        if self._config.cuda_graph_expert:
            model.action_model.predict_action = _GraphedFlowSampler(model.action_model.predict_action)
        self.model = model

    def _verify_run(self, starvla_config: DictConfig, norm_stats: dict[str, Any], bundle_dir: Path) -> None:
        """Refuses a starVLA run whose input or output layout this policy's config does not mirror."""
        framework = starvla_config.framework
        vla_data = starvla_config.datasets.vla_data
        sampler: dict[str, Any] = norm_stats.get("sampler", {})
        config = self._config
        problems: list[str] = []
        if framework.name != "QwenDrivePI":
            problems.append(f"framework {framework.name} is not QwenDrivePI")
        eye_gate = framework.get("eye_gate", {})
        if eye_gate.get("enabled", False) and str(eye_gate.get("scheme", "standard")) == "vlm_keep":
            problems.append("eye gate scheme vlm_keep (its own predict path) is not served")
        layout = (vla_data.get("state_mode", "drivemoe"), vla_data.get("action_mode", "waypoints"))
        action_mean_std = norm_stats["action"].get("norm", "q01_q99") == "mean_std"
        if layout == ("drivemoe", "waypoints"):
            if vla_data.get("target_point_in_prompt", False) or vla_data.get("speed_in_prompt", False):
                problems.append("drivemoe prompts with the target point or speed")
            expected_names = (_STATE_NAMES, ["x", "y"])
            if action_mean_std:
                problems.append("mean / std waypoint normalization")
        elif layout == ("traj", "unicycle"):
            with_target = bool(vla_data.get("state_target_point", True))
            expected_names = (_TRAJ_STATE_NAMES + (_TARGET_STATE_NAMES if with_target else []), _UNICYCLE_ACTION_NAMES)
            if not action_mean_std:
                problems.append("q01 / q99 unicycle normalization")
        else:
            problems.append(f"state_mode {layout[0]} / action_mode {layout[1]}")
            expected_names = (norm_stats["state"]["names"], norm_stats["action"]["names"])
        if norm_stats["state"]["names"] != expected_names[0] or norm_stats["action"]["names"] != expected_names[1]:
            problems.append(f"state {norm_stats['state']['names']} / action {norm_stats['action']['names']}")

        dt_us = round(float(sampler.get("dt", 0.1)) * 1e6)
        action_horizon = int(framework.action_model.action_horizon)
        # Waypoints include t0 itself, controls start at t0 + dt.
        future_steps = action_horizon - 1 if layout[1] == "waypoints" else action_horizon
        if _anchors_enabled(starvla_config):
            problems += _anchor_problems(bundle_dir / config.anchors_file, norm_stats, action_horizon, layout[1])
        image_offsets_us = sorted(round(float(offset) * 1e6) for offset in sampler.get("image_offsets", [0.1, 0.0]))
        expected: dict[str, tuple[Any, Any]] = {
            "trajectory_interval_us": (config.trajectory_interval_us, dt_us),
            "trajectory_horizon_us": (config.trajectory_horizon_us, future_steps * dt_us),
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
        """
        Compiles the linear-attention kernels and captures the expert graph at every served input shape: the
        prompt's numbers change its token count, so each count gets its graph before the first rollout rather
        than a capture stall (and a memory step) in the middle of one.
        """
        (current_width, current_height), (history_width, history_height) = self._current_size, self._history_size
        state_names = _STATE_NAMES if self._state_mode == "drivemoe" else _TRAJ_STATE_NAMES
        if self._state_mode == "traj" and self._state_target_point:
            state_names = state_names + _TARGET_STATE_NAMES
        images = [Image.new("RGB", (history_width, history_height))] * self._config.num_history_frames + [
            Image.new("RGB", (current_width, current_height)),
        ]
        state = np.zeros((self._config.state_horizon, len(state_names)), dtype=np.float32)
        examples = [{"image": images, "state": state, **inputs} for inputs in self._prompt_inputs_per_length()]
        try:
            self.model.predict_action([examples[0]])
        except Exception:
            if self._config.linear_attention_kernels != "auto":
                raise
            LOG.exception("the linear-attention Triton kernels failed; falling back to the torch implementation")
            _use_torch_linear_attention()
        for example in examples:
            for _ in range(2):
                self.model.predict_action([example])
        sampler = self.model.action_model.predict_action
        LOG.info(
            f"warmed up {len(examples)} prompt lengths, "
            f"{sampler.num_graphs if isinstance(sampler, _GraphedFlowSampler) else 0} expert graphs, "
            f"{torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else 0.0:.2f} GiB reserved at peak",
        )

    def _prompt_inputs_per_length(self) -> list[dict[str, Any]]:
        """One set of prompt inputs per prompt token count a drive can produce (speed and target point digits)."""
        if not (self._target_point_in_prompt or self._speed_in_prompt):
            return [self._prompt_inputs(np.array([25.0, 0.0], dtype=np.float32), 10.0)]
        tokenizer = self.model.qwen_vl_interface.processor.tokenizer
        by_length: dict[int, dict[str, Any]] = {}
        # Signs and one or two integer digits; 25 m targets and highway speeds stay below 100.
        for speed in (0.0, 10.0, 35.0):
            for target_x in (-10.0, -1.0, 1.0, 10.0, 25.0):
                for target_y in (-10.0, -1.0, 1.0, 10.0):
                    inputs = self._prompt_inputs(np.array([target_x, target_y], dtype=np.float32), speed)
                    by_length.setdefault(len(tokenizer(inputs["lang"])["input_ids"]), inputs)
        return [by_length[length] for length in sorted(by_length)]

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
            ego_state=torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
                _ego_dynamics(scene_api, config.state_horizon)
                if self._state_mode == "drivemoe"
                else _traj_state(scene_api, config.state_horizon, config.trajectory_interval_s),
            ),
            target_point=torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
                _straight_line_target_point(scene_api, config.required_target_point_distances_m[0]),
            ),
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
        ego_states = cast(
            "npt.NDArray[np.float32]",
            features.ego_state.float().cpu().numpy(),  # pyright: ignore[reportUnknownMemberType]
        )
        # The policy's own target point, placed as in training; the navigation one runs along the route's arc.
        del navigation
        target_xy = cast(
            "npt.NDArray[np.float32]",
            features.target_point.float().cpu().numpy(),  # pyright: ignore[reportUnknownMemberType]
        )
        # The current speed: the anchor row's speed, the v0 the training loader read at t0.
        v0 = ego_states[:, -1, _SPEED_COLUMN[self._state_mode]].astype(np.float64)

        examples: list[dict[str, Any]] = []
        for index in range(current_frames.shape[0]):
            state = ego_states[index]
            if self._state_target_point:
                state = np.concatenate([state, np.repeat(target_xy[index][None], state.shape[0], axis=0)], axis=1)
            examples.append(
                {
                    "image": [Image.fromarray(frame) for frame in history_frames[index]]
                    + [Image.fromarray(current_frames[index])],
                    "state": _normalize(state, self._state_q01, self._state_q99),
                    **self._prompt_inputs(target_xy[index], float(v0[index])),
                },
            )
        output = self.model.predict_action(examples)
        normalized_actions = cast("npt.NDArray[np.float32]", output["normalized_actions"])
        anchor_reference_xy: npt.NDArray[np.float32] | None = None
        if self._unicycle_dt_s is None:
            # The action starts at t0 itself; the trajectory grid starts one interval later.
            waypoints_xy = _unnormalize(normalized_actions, self._action_low, self._action_high)[:, 1:]
        else:
            controls = _unstandardize(normalized_actions, self._action_low, self._action_high)
            # Positions at t0 + dt .. t0 + N·dt, integrated from the current speed as starVLA's action_to_xy.
            waypoints_xy = _rollout_unicycle(controls, v0, self._unicycle_dt_s)
        if "anchor_reference" in output:
            # Control-space runs return the reference integrated from v0 with t0 itself first.
            reference = np.asarray(output["anchor_reference"], dtype=np.float32)
            anchor_reference_xy = reference[:, 1:] if reference.shape[1] > waypoints_xy.shape[1] else reference
        device = features.ego_state.device
        return QwenDrivePIPredictions(
            waypoints_xy=torch.as_tensor(waypoints_xy, dtype=torch.float32, device=device),
            anchor_reference_xy=(
                torch.as_tensor(anchor_reference_xy, dtype=torch.float32, device=device)
                if anchor_reference_xy is not None
                else None
            ),
        )

    def _prompt_inputs(self, target_xy: npt.NDArray[np.float32], v0: float) -> dict[str, Any]:
        """
        The example keys beyond image and state: the prompt with the target point and the speed when the run
        put them there (starVLA's drive_prompt, byte for byte), and v0, which control-space anchors integrate from.
        """
        # target_xy: the eye gate's query (eye_gate.query target / target_speed / target_state) reads it raw.
        inputs: dict[str, Any] = {"v0": v0, "target_xy": np.asarray(target_xy, dtype=np.float32)}
        if self._target_point_in_prompt or self._speed_in_prompt:
            inputs["lang"] = _drive_prompt(
                self._prompt,
                target_xy if self._target_point_in_prompt else None,
                v0 if self._speed_in_prompt else None,
            )
        return inputs

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

    @override
    def visualize_batch(
        self,
        features: QwenDrivePIFeatures,
        labels: QwenDrivePILabels | None,
        navigation: NavigationConditioning,
        predictions: QwenDrivePIPredictions,
        scene_api: SceneAPI,
    ) -> dict[str, torch.Tensor]:
        """
        The first sample's current frame next to VaVAM's BEV sheet: the predicted waypoints, the target point,
        the ego history (traj runs), the anchor reference (AnchorVLA runs) and the logged future when known.
        """
        ego_state = scene_api.get_ego_state_se3_at_iteration(0)
        assert ego_state is not None, "Ego state should be available for visualization!"
        ego_box = ego_state.bounding_box_se2.array.copy()
        ego_box[BoundingBoxSE2Index.SE2] = abs_to_rel_se2_array(
            origin=ego_state.rear_axle_se2,
            pose_se2_array=ego_box[None, BoundingBoxSE2Index.SE2],
        )[0]
        ego_states = features.ego_state[0].float().cpu().numpy()  # pyright: ignore[reportUnknownMemberType]
        extra_trajectories: list[tuple[str, npt.NDArray[np.float32], TrajectoryStyleConfig]] = []
        if self._state_mode == "traj":
            extra_trajectories.append(("history", ego_states[:, :2], _PAST_TRAJECTORY_STYLE))
        if predictions.anchor_reference_xy is not None:
            extra_trajectories.append(
                ("anchor reference", _first_sample(predictions.anchor_reference_xy), _ANCHOR_REFERENCE_STYLE),
            )
        bev = render_bev(
            _first_sample(predictions.waypoints_xy),
            _first_sample(labels.trajectory_xy) if labels is not None else None,
            _first_sample(features.target_point[:, None]),
            ego_box.astype(np.float32),
            self._config.visualization_config,
            extra_trajectories,
        )

        camera = np.ascontiguousarray(features.current_frame[0].cpu().numpy())  # pyright: ignore[reportUnknownMemberType]
        speed = float(ego_states[-1, _SPEED_COLUMN[self._state_mode]])
        end_x, end_y = (float(value) for value in _first_sample(predictions.waypoints_xy)[-1])
        caption = (
            f"t {ego_state.timestamp.time_us / 1e6:7.1f} s   v {speed:5.2f} m/s   "
            f"plan end x {end_x:5.1f} m  y {end_y:+5.1f} m"
        )
        cv2.rectangle(camera, (0, 0), (camera.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(camera, caption, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        height = camera.shape[0]
        scaled_bev = cv2.resize(bev, (round(bev.shape[1] * height / bev.shape[0]), height))
        return {"drive": torch.from_numpy(np.concatenate([camera, scaled_bev], axis=1))}  # pyright: ignore[reportUnknownMemberType]


def _first_sample(tensor: torch.Tensor) -> npt.NDArray[np.float32]:
    """First sample of a batched tensor as a float32 numpy array."""
    return cast(
        "npt.NDArray[np.float32]",
        tensor[0].detach().float().cpu().numpy(),  # pyright: ignore[reportUnknownMemberType]
    )


def _anchors_enabled(starvla_config: DictConfig) -> bool:
    return bool(starvla_config.framework.get("anchor", {}).get("enabled", False))


def _anchor_problems(
    anchors_file: Path,
    norm_stats: dict[str, Any],
    action_horizon: int,
    action_mode: str,
) -> list[str]:
    """What keeps a bundled anchor codebook from matching the run: its presence, space, horizon and statistics."""
    if not anchors_file.is_file():
        return [f"the AnchorVLA run needs its anchor codebook at {anchors_file}"]
    codebook = np.load(anchors_file)
    problems: list[str] = []
    space = str(codebook["space"]) if "space" in codebook.files else "waypoint"
    if (space == "control") != (action_mode == "unicycle"):
        problems.append(f"{space}-space anchors with action_mode {action_mode}")
    if codebook["anchors"].shape[1] != action_horizon:
        problems.append(f"anchors of {codebook['anchors'].shape[1]} steps vs. action_horizon {action_horizon}")
    action_stats = norm_stats["action"]
    if "action_mean" in codebook.files and "mean" in action_stats:
        for key, stat in (("action_mean", "mean"), ("action_std", "std")):
            if not np.allclose(codebook[key], np.asarray(action_stats[stat]), rtol=1e-4, atol=1e-7):
                problems.append(f"the codebook's {key} differs from the run's action {stat}")
    return problems


def _use_torch_linear_attention() -> None:
    """Points Qwen3.5's gated-delta-rule calls back at transformers' torch implementations."""
    from transformers.models.qwen3_5 import modeling_qwen3_5

    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        setattr(modeling_qwen3_5, name, inspect.unwrap(getattr(modeling_qwen3_5, name)))


class _NoAttentionProbe:
    """
    Stands in for starVLA's CrossAttentionProbe: its hooks record the DiT's attention for eval figures only,
    cost a q·k product per block and record nothing under a replayed CUDA graph.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def __enter__(self) -> _NoAttentionProbe:
        return self

    def __exit__(self, *exc: object) -> None:
        del exc

    def mean(self) -> None:
        return None


def _disable_evaluation_extras(model: Any) -> None:
    """
    Turns off what a run's eval settings add to every prediction (drive.diagnostics: probes and a black-image
    re-run; eye_gate.eval_controls: soft-mask control predictions; the eye gate's attention maps), none of which
    the served plan uses. The modules stay built, so the checkpoint still loads strictly.
    """
    from starVLA.model.framework.VLM4A import QwenDrivePI as qwendrive_pi_module  # pyright: ignore[reportMissingImports]

    model.diagnostics = False
    with open_dict(model.config):
        model.config.framework.eye_gate.eval_controls = False
    qwendrive_pi_module.CrossAttentionProbe = _NoAttentionProbe
    model._add_attention_outputs = lambda *args, **kwargs: None


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


class _AveragedFlowSampler:
    """
    The mean of n flow-matching samples per plan: every input is repeated n times along the batch, so the
    n denoising loops run as one batched pass (and one CUDA graph), and the samples average per example.
    A given noise (paired eval controls) already fixes the sample and runs once.
    """

    def __init__(self, sample: Any, num_samples: int) -> None:
        self._sample = sample
        self._num_samples = num_samples

    def __call__(
        self,
        vl_embs_list: list[torch.Tensor],
        state: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        query_bias: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is not None:
            return self._sample(
                vl_embs_list,
                state,
                encoder_attention_mask=encoder_attention_mask,
                query_bias=query_bias,
                noise=noise,
            )
        batch_size = vl_embs_list[0].shape[0]
        # The list repeats one tensor for the blocks that share a VLM layer; each is repeated once.
        repeated: dict[int, torch.Tensor] = {}
        for tensor in vl_embs_list:
            repeated.setdefault(id(tensor), self._repeat(tensor))
        actions = self._sample(
            [repeated[id(tensor)] for tensor in vl_embs_list],
            self._repeat(state),
            encoder_attention_mask=self._repeat(encoder_attention_mask),
            query_bias=self._repeat(query_bias),
        )
        return actions.unflatten(0, (batch_size, self._num_samples)).mean(dim=1)

    def _repeat(self, tensor: torch.Tensor | None) -> Any:
        return tensor.repeat_interleave(self._num_samples, dim=0) if tensor is not None else None


class _GraphedFlowSampler:
    """
    The expert's whole denoising loop replayed from one CUDA graph per input shape: the eager loop
    launches thousands of small kernels per plan, so its run time is launch overhead. AnchorVLA's
    query bias is one more static input. Every graph draws from one memory pool: plans replay one
    graph at a time (the driver serializes inference), so the graphs' temporaries never overlap.
    """

    def __init__(self, sample: Any) -> None:
        self._sample = sample
        self._graphs: dict[tuple[Any, ...], tuple[torch.cuda.CUDAGraph, dict[int, torch.Tensor], torch.Tensor]] = {}
        self._pool: Any = None

    @property
    def num_graphs(self) -> int:
        return len(self._graphs)

    def __call__(
        self,
        vl_embs_list: list[torch.Tensor],
        state: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        query_bias: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # A given noise (paired eval controls, regression heads) runs eagerly; the graph draws its own.
        if noise is not None:
            return self._sample(
                vl_embs_list,
                state,
                encoder_attention_mask=encoder_attention_mask,
                query_bias=query_bias,
                noise=noise,
            )
        if state is None or encoder_attention_mask is None:
            return self._sample(
                vl_embs_list, state, encoder_attention_mask=encoder_attention_mask, query_bias=query_bias
            )
        tail = [state, encoder_attention_mask] if query_bias is None else [state, encoder_attention_mask, query_bias]
        inputs = [*vl_embs_list, *tail]
        if not all(tensor.is_cuda for tensor in inputs):
            return self._sample(
                vl_embs_list, state, encoder_attention_mask=encoder_attention_mask, query_bias=query_bias
            )
        # The list repeats one tensor for the blocks that share a VLM layer; the graph keeps that aliasing.
        positions = [next(index for index, other in enumerate(inputs) if other is tensor) for tensor in inputs]
        key = (len(tail), tuple(positions), *((tensor.shape, tensor.dtype) for tensor in inputs))
        if key not in self._graphs:
            self._graphs[key] = self._capture(inputs, positions, len(tail))
        graph, static_inputs, static_output = self._graphs[key]
        for position, static in static_inputs.items():
            static.copy_(inputs[position])
        graph.replay()
        return static_output.clone()

    def _capture(
        self,
        inputs: list[torch.Tensor],
        positions: list[int],
        num_tail: int,
    ) -> tuple[torch.cuda.CUDAGraph, dict[int, torch.Tensor], torch.Tensor]:
        static_inputs = {position: inputs[position].clone() for position in sorted(set(positions))}
        static_vl_embs = [static_inputs[position] for position in positions[:-num_tail]]
        state, mask, *query_bias = (static_inputs[position] for position in positions[-num_tail:])

        def sample() -> torch.Tensor:
            return self._sample(
                static_vl_embs,
                state,
                encoder_attention_mask=mask,
                query_bias=query_bias[0] if query_bias else None,
            )

        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())  # pyright: ignore[reportUnknownMemberType]
        with torch.cuda.stream(side_stream):
            for _ in range(2):
                sample()
        torch.cuda.current_stream().wait_stream(side_stream)  # pyright: ignore[reportUnknownMemberType]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            static_output = sample()
        if self._pool is None:
            self._pool = graph.pool()
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


def _traj_state(scene_api: SceneAPI, state_horizon: int, interval_s: float) -> npt.NDArray[np.float32]:
    """
    Per-step [x, y, heading relative to the anchor, speed, longitudinal acceleration] at the anchor and the
    state_horizon - 1 iterations before it, oldest first, positions in the anchor's rear-axle frame (x forward,
    y left) as the starVLA traj state. A rollout younger than that extends its oldest state backwards at constant
    velocity, so the padded rows stay a consistent motion (repeating it would place a moving ego at rest).
    """
    anchor = scene_api.get_ego_state_se3_at_iteration(0)
    assert anchor is not None, "Ego state should be available for feature computation!"
    anchor_pose = anchor.rear_axle_se2
    anchor_yaw = float(anchor_pose.yaw)
    cos_anchor, sin_anchor = math.cos(anchor_yaw), math.sin(anchor_yaw)

    rows: list[npt.NDArray[np.float32]] = []
    oldest: EgoStateSE3 = anchor
    missing_steps = 0
    for iteration in range(0, -state_horizon, -1):
        ego_state = scene_api.get_ego_state_se3_at_iteration(iteration)
        if ego_state is not None and missing_steps == 0:
            oldest = ego_state
        else:
            missing_steps += 1
        dynamic_state = oldest.dynamic_state_se3
        assert dynamic_state is not None, "Ego dynamic state should be available for feature computation!"
        pose = oldest.rear_axle_se2
        speed = float(np.linalg.norm(dynamic_state.velocity_2d.array))
        back_m = speed * interval_s * missing_steps
        dx = float(pose.x) - back_m * math.cos(float(pose.yaw)) - float(anchor_pose.x)
        dy = float(pose.y) - back_m * math.sin(float(pose.yaw)) - float(anchor_pose.y)
        rows.append(
            np.array(
                [
                    cos_anchor * dx + sin_anchor * dy,
                    -sin_anchor * dx + cos_anchor * dy,
                    math.remainder(float(pose.yaw) - anchor_yaw, math.tau),
                    speed,
                    # The simulators report the rig-frame acceleration, whose x is the longitudinal one.
                    float(dynamic_state.acceleration_3d.array[0]),
                ],
                dtype=np.float32,
            ),
        )
    return np.stack(rows[::-1])


def _straight_line_target_point(scene_api: SceneAPI, distance_m: float) -> npt.NDArray[np.float32]:
    """
    The first route point ahead of the ego at least distance_m away in a straight line, or the route's last
    point, in the anchor's rear-axle frame: starVLA's EgoTrack.target_point, which takes the first logged future
    position that far, with the route standing in for the future (the navigation target runs along the arc).
    """
    anchor = scene_api.get_ego_state_se3_at_iteration(0)
    route = scene_api.get_route()
    progress_m = scene_api.get_route_progress_at_iteration(0)
    if anchor is None or route is None or progress_m is None:
        raise InsufficientRouteError("QwenDrivePI places its target point on the route, but the scene has none.")
    _, polyline_arc_m, polyline_xyz = route
    ahead_xy = np.asarray(polyline_xyz, dtype=np.float64)[np.asarray(polyline_arc_m) >= progress_m, :2]
    if len(ahead_xy) == 0:
        ahead_xy = np.asarray(polyline_xyz, dtype=np.float64)[-1:, :2]
    relative = abs_to_rel_se2_array(
        origin=anchor.rear_axle_se2,
        pose_se2_array=np.concatenate([ahead_xy, np.zeros((len(ahead_xy), 1))], axis=1),
    )[:, :2]
    far = np.nonzero(np.linalg.norm(relative, axis=1) >= distance_m)[0]
    return relative[far[0] if len(far) else -1].astype(np.float32)


def _rollout_unicycle(
    controls: npt.NDArray[np.float32],
    v0: npt.NDArray[np.float64],
    dt: float,
) -> npt.NDArray[np.float32]:
    """
    Positions (B, N, 2) at t0 + dt .. t0 + N·dt from raw [acceleration, curvature] controls (B, N, 2) and the
    current speeds (B,): starVLA's unicycle_rollout (Alpamayo's action_to_traj), except that the speed stops at
    zero. A braking plan then ends at rest instead of rolling backwards, which would flip the served headings
    (derived from the positions) by pi; while the speed stays positive the positions are identical.
    """
    controls64 = np.asarray(controls, dtype=np.float64)
    accel, kappa = controls64[..., 0], controls64[..., 1]
    # The model's speed profile, clipped at zero only where it is read: a dip below zero that returns to zero
    # (the training targets' fit overshoots at every stop) stays at rest instead of driving off again.
    v = np.maximum(np.concatenate([v0[:, None], v0[:, None] + np.cumsum(accel * dt, axis=-1)], axis=-1), 0.0)
    accel = np.diff(v, axis=-1) / dt
    dtheta = kappa * v[..., :-1] * dt + kappa * accel * (0.5 * dt**2)
    theta = np.concatenate([np.zeros_like(v0)[:, None], np.cumsum(dtheta, axis=-1)], axis=-1)
    vx, vy = v * np.cos(theta), v * np.sin(theta)
    x = np.cumsum((vx[..., :-1] + vx[..., 1:]) * (0.5 * dt), axis=-1)
    y = np.cumsum((vy[..., :-1] + vy[..., 1:]) * (0.5 * dt), axis=-1)
    return np.stack([x, y], axis=-1).astype(np.float32)


def _drive_prompt(prompt: str, target_xy: npt.NDArray[np.float32] | None, speed: float | None) -> str:
    """
    starVLA's pai_av_drive.drive_prompt verbatim; importing it would pull starVLA.dataloader's training-only
    dependencies (accelerate) into the driver.
    """
    parts = [prompt]
    if speed is not None:
        parts.append(f"speed: {float(speed):.1f} m/s")
    if target_xy is not None:
        x, y = (float(v) for v in target_xy)
        parts.append(f"target point: x {x:.1f} m, y {y:.1f} m")
    return ". ".join(parts)


def _unstandardize(
    values: npt.NDArray[np.float32],
    mean: npt.NDArray[np.float32],
    std: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """z-score -> raw, as pai_av_drive.Normalizer for statistics marked mean_std."""
    return (np.asarray(values, np.float32) * std + mean).astype(np.float32)


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
