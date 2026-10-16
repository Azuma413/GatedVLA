from __future__ import annotations

from dataclasses import dataclass, field

from py123d.datatypes import CameraID
from typing_extensions import override

from py123d_garage.api.abstract_policy_config import AbstractPolicyConfig
from py123d_garage.datatypes.numerics import NonNegativeInt, PositiveInt


@dataclass
class QwenDrivePIConfig(AbstractPolicyConfig):
    """
    starVLA's QwenDrivePI (Qwen-Drive VLM + flow-matching expert) trained on PAI-AV, evaluated as trained.

    The architecture, the normalization and the input layout come from the starVLA run files bundled next
    to the checkpoint (scripts/pretrained/qwendrive_pi.sh); the values below mirror them and are verified
    against them when the model is built.
    """

    # -- Values --

    # The h25 runs predict 26 waypoints t0 .. t0 + 2.5 s at 10 Hz; t0 itself is dropped.
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

    # Closed-loop drivers query the model every n-th trajectory interval and serve the cached plan in
    # between: one inference takes longer than the 100 ms AlpaSim tick (see the challenge's 0.1 s target).
    replan_every_n_intervals: PositiveInt = 2

    # Qwen3.5's linear-attention layers: "fla" runs flash-linear-attention's Triton kernels, "torch" the
    # reference implementation (about 50 ms slower per plan), "auto" falls back to torch when the Triton
    # kernels fail to compile or load at warmup (e.g. a noexec /tmp in the challenge sandbox).
    linear_attention_kernels: str = "auto"

    # Replays the expert's denoising loop from a CUDA graph; same numerics, a fraction of the launch overhead.
    cuda_graph_expert: bool = True

    # -- Bundle files, relative to the directory of policy_config.evaluation_checkpoint_file --

    # The starVLA run config.yaml of the checkpoint.
    starvla_config_file: str = "starvla_config.yaml"
    # The q01 / q99 state and action statistics of the run.
    norm_stats_file: str = "norm_stats.json"
    # Holds <name of framework.qwenvl.base_vlm>/ with its config, tokenizer and processor files; no weights.
    base_vlm_dir: str = "base_vlm"

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
