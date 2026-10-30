from __future__ import annotations

from dataclasses import dataclass, field

from py123d.datatypes import CameraID
from typing_extensions import override

from py123d_garage.api.abstract_policy_config import AbstractPolicyConfig
from py123d_garage.config.schema.policy.vavam_config import VavamVisualizationConfig
from py123d_garage.datatypes.numerics import NonNegativeInt, PositiveInt


@dataclass
class QwenDrivePIConfig(AbstractPolicyConfig):
    """
    starVLA's QwenDrivePI (Qwen-Drive VLM + flow-matching expert) trained on PAI-AV, evaluated as trained.

    The architecture, the normalization and the input layout come from the starVLA run files bundled next
    to the checkpoint (scripts/pretrained/qwendrive_pi.sh); the values below mirror them and are verified
    against them when the model is built.

    Two run layouts are served, read from the run's datasets.vla_data:
    - drivemoe / waypoints (h25): per-step dynamics + target point as state, 26 waypoints t0 .. t0 + 2.5 s.
    - traj / unicycle (v2, AnchorVLA v2): past ego poses, speed and acc_lon as state, the target point and the
      speed in the prompt, 25 [acceleration, curvature] controls integrated from the current speed.
    """

    # -- Values --

    # Both layouts serve 25 positions t0 + 0.1 .. t0 + 2.5 s at 10 Hz (the waypoint layout drops t0 itself).
    trajectory_horizon_us: PositiveInt = 2_500_000
    trajectory_interval_us: PositiveInt = 100_000

    # Training places the target point at the first future position 25 m away.
    required_target_point_distances_m: list[float] = field(default_factory=lambda: [25.0])

    input_camera: dict[str, CameraID] = field(
        default_factory=lambda: {
            "physical-ai-av": CameraID.FTCAM_F0,
        },
    )

    # Context frames before the current one, spaced by context_frame_interval_us (training: t0 - 0.1 s).
    num_history_frames: int = 1
    context_frame_interval_us: PositiveInt = 100_000

    # Ego states t0 - (state_horizon - 1) * 0.1 s .. t0 the flow-matching expert reads.
    state_horizon: int = 5

    # The PAI-AV recording size the training images were resized from; renders of other widths are
    # resized to the same token grid.
    source_image_width: int = 1920
    source_image_height: int = 1080

    # Flow-matching integration steps; None keeps the trained framework.action_model.num_inference_timesteps.
    num_inference_timesteps: int | None = None

    # Flow-matching samples drawn per plan (one batched expert pass) and averaged in the normalized action
    # space; 1 serves a single sample, as training's evaluation does.
    num_action_samples: PositiveInt = 1

    # Closed-loop drivers query the model every n-th trajectory interval and serve the cached plan in
    # between: one inference takes longer than the 100 ms AlpaSim tick (see the challenge's 0.1 s target).
    # Every 0.5 s keeps the mean Drive call under 0.1 s for inferences up to 0.25 s with two rollouts
    # sharing one driver, as the challenge's EC2 preset runs them.
    replan_every_n_intervals: PositiveInt = 5

    # Qwen3.5's linear-attention layers: "fla" runs flash-linear-attention's Triton kernels, "torch" the
    # reference implementation (about 50 ms slower per plan), "auto" falls back to torch when the Triton
    # kernels fail to compile or load at warmup (e.g. a noexec /tmp in the challenge sandbox).
    linear_attention_kernels: str = "auto"

    # Replays the expert's denoising loop from a CUDA graph; same numerics, a fraction of the launch overhead.
    cuda_graph_expert: bool = True

    # -- Bundle files, relative to the directory of policy_config.evaluation_checkpoint_file --

    # The starVLA run config.yaml of the checkpoint.
    starvla_config_file: str = "starvla_config.yaml"
    # The state (q01 / q99) and action (q01 / q99, or mean / std for unicycle controls) statistics of the run.
    norm_stats_file: str = "norm_stats.json"
    # Holds <name of framework.qwenvl.base_vlm>/ with its config, tokenizer and processor files; no weights.
    base_vlm_dir: str = "base_vlm"
    # The anchor codebook of an AnchorVLA run (framework.anchor.path); the model registers it as non-persistent
    # buffers, so it is not in model.safetensors.
    anchors_file: str = "anchors.npz"

    # The drive video's BEV sheet, drawn next to the camera frame: 2.5 s at 24 m/s ahead, the history behind,
    # and the 25 m target point of a turn to either side.
    visualization_config: VavamVisualizationConfig = field(
        default_factory=lambda: VavamVisualizationConfig(
            bev_min_x_m=-10.0,
            bev_max_x_m=60.0,
            bev_half_width_m=30.0,
            bev_pixels_per_meter=10.0,
        ),
    )

    @property
    @override
    def required_cameras(self) -> dict[str, list[CameraID]]:
        return {dataset: [camera] for dataset, camera in self.input_camera.items()}

    @property
    @override
    def required_history_duration_us(self) -> NonNegativeInt:
        return max(
            self.num_history_frames * self.context_frame_interval_us,
            (self.state_horizon - 1) * self.trajectory_interval_us,
        )

    @property
    @override
    def required_past_camera_interval_us(self) -> PositiveInt:
        return self.context_frame_interval_us

    @property
    @override
    def replan_interval_us(self) -> PositiveInt:
        return self.replan_every_n_intervals * self.trajectory_interval_us

    @property
    @override
    def required_past_ego_state_interval_us(self) -> PositiveInt:
        return self.trajectory_interval_us
