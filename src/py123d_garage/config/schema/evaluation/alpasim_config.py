from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import override

from py123d_garage.api.abstract_benchmark_config import AbstractBenchmarkConfig
from py123d_garage.config.schema.policy.policy_config import EvaluationPolicyConfig
from py123d_garage.datatypes.numerics import NonNegativeFloat, PositiveFloat, PositiveInt


@dataclass
class EgoVehicleConfig:
    # Vehicle name written into the ego metadata.
    vehicle_name: str = "alpasim"
    # Bounding-box width of the ego vehicle.
    width_m: float = 2.297
    # Bounding-box length of the ego vehicle.
    length_m: float = 5.176
    # Bounding-box height of the ego vehicle.
    height_m: float = 1.777
    # Distance between the front and rear axles.
    wheel_base_m: float = 3.089
    # Longitudinal offset from the rear axle to the box center.
    rear_axle_to_center_longitudinal_m: float = 1.461


@dataclass
class AebConfig:
    """Camera-based emergency braking on top of the served plans."""

    # Directory of a transformers RT-DETR checkpoint; None serves the plans as predicted.
    detector_dir: str | None = None
    # Detection confidence a road user needs.
    score_threshold: float = 0.5
    # Square side the front frame is resized to for the detector.
    input_size: int = 640
    # Least time between two detections of a session; 0 looks at every front frame.
    detection_interval_s: float = 0.0
    # Fraction cut off each end of a box's bottom edge: the corners of an obliquely seen road user lie beside it.
    edge_shrink: float = 0.3
    # Half width of the ego plus the lateral margin of its corridor around the planned path.
    corridor_half_width_m: float = 1.3
    # Rear axle to front bumper.
    front_overhang_m: float = 4.05
    # Standstill gap to the obstacle.
    stop_gap_m: float = 2.5
    # Reaction delay of the controller, during which the closing speed still eats the gap.
    latency_s: float = 0.4
    # Brake once stopping in time needs more than this deceleration.
    trigger_decel_mps2: float = 3.0
    # Hardest braking the re-timed plan asks for.
    max_decel_mps2: float = 8.0
    # No acceleration while the time to collision is below this.
    no_accel_ttc_s: float = 3.0
    # Distance samples of the lead obstacle the closing speed is fit over.
    track_window_s: float = 1.0
    # Samples the closing speed needs before the rule acts.
    min_track_samples: int = 6
    # A jump larger than this in the lead distance restarts its track.
    track_jump_m: float = 4.0


@dataclass
class AlpasimBenchmarkConfig(AbstractBenchmarkConfig):
    # -- Config objects --

    # The policy under evaluation; evaluation_checkpoint_file selects its weights.
    policy_config: EvaluationPolicyConfig = field(default_factory=EvaluationPolicyConfig)
    # Ego geometry the driver builds its metadata from.
    ego_vehicle_config: EgoVehicleConfig = field(default_factory=EgoVehicleConfig)
    # Emergency braking for road users ahead that the policy misses.
    aeb_config: AebConfig = field(default_factory=AebConfig)

    # -- Atomic settings --

    # Git hash of the code baked into the submission image.
    git_hash: str = "unknown"

    # Inference device of the driver process.
    device: str = "cuda"
    # Bind address of the gRPC driver service.
    host: str = "0.0.0.0"
    # Port of the gRPC driver service.
    port: int = 6789
    # gRPC server worker threads.
    max_workers: int = 8
    # Writes a drive video every n-th plan of every session into <run dir>/views, played in real time; 0 disables
    # the recording.
    visualization_interval: int = 0
    # Spacing the route polyline is resampled at.
    route_resolution_m: float = 0.1
    # ACT's temporal ensembling on the served plans: each new plan is averaged with the previous
    # temporal_ensemble_plans - 1 predictions where they overlap in time, each re-anchored at the current pose and
    # weighted exp(-temporal_ensemble_decay * its age in plans); 1 serves every plan as predicted.
    temporal_ensemble_plans: PositiveInt = 1
    temporal_ensemble_decay: NonNegativeFloat = 0.0

    # -- The AlpaSim protocol, hard-coded --

    @property
    @override
    def required_trajectory_horizon_us(self) -> PositiveInt:
        """Covers the 500 ms replan interval plus the MPC's 2 s reference lookahead."""
        return 2_500_000

    @property
    @override
    def min_served_target_point_distance_m(self) -> NonNegativeFloat:
        """Nearest target point the served route geometry supports."""
        return 20.0

    @property
    @override
    def max_served_target_point_distance_m(self) -> PositiveFloat:
        """Farthest target point the served route geometry supports."""
        return 80.0

    @property
    @override
    def served_ego_state_interval_us(self) -> PositiveInt:
        """The simulator drives at 10 Hz ticks."""
        return 100_000

    @property
    @override
    def served_camera_interval_us(self) -> PositiveInt:
        """Camera frames arrive once per drive tick."""
        return 100_000

    @property
    @override
    def served_lidar_interval_us(self) -> PositiveInt | None:
        """Alpasim does not have lidar."""
        return None
