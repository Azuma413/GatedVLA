import json
import logging
import os
import shutil
from collections import deque
from copy import deepcopy

import carla
import cv2
import numpy as np
import torch
from agents.navigation.local_planner import RoadOption
from beartype import beartype
from leaderboard.autoagents import autonomous_agent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from lead.common import common_utils
from lead.common.base_agent import BaseAgent
from lead.common.constants import TargetDataset
from lead.common.pid_controller import PIDController
from lead.common.route_planner import RoutePlanner
from lead.common.sensor_setup import av_sensor_setup
from lead.inference.closed_loop_inference import ClosedLoopInference
from lead.inference.config_closed_loop import ClosedLoopConfig
from lead.inference.infraction_recorder import InfractionRecorder
from lead.inference.video_recorder import VideoRecorder
from lead.inference.vlajepa_client import VLAJEPAClient
from lead.training.config_training import TrainingConfig

LOG = logging.getLogger(__name__)


def get_entry_point():
    return "VLAJEPAAgent"


class VLAJEPAEvalConfig(TrainingConfig):
    """LEAD sensor config that fixes the online rig to the 4 Alpamayo cameras.

    ``data_save_freq = 2`` means predicted waypoints are spaced at
    ``data_save_freq / carla_fps = 0.1s`` (matching the training frame rate), so
    ``execute_waypoints`` reads the 0.5s / 1.0s marks at indices 4 / 9.
    """

    data_save_freq = 2

    @property
    def target_dataset(self):
        return TargetDataset.CARLA_ALPAMAYO_4CAMERAS

    @property
    def num_available_cameras(self):
        return 4

    @property
    def used_cameras(self):
        return [True, True, True, True]


class VLAJEPAAgent(BaseAgent, autonomous_agent.AutonomousAgent):
    @beartype
    def setup(self, path_to_conf_file: str, _=None, __=None):
        self.config_closed_loop = ClosedLoopConfig()
        super().setup(sensor_agent=True)
        self.config_path = path_to_conf_file.split("+")[0]
        self.step = -1
        self.initialized = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.training_config = VLAJEPAEvalConfig({"target_dataset": 7})
        self.closed_loop_controller = self._build_waypoint_controller()

        server_host = os.getenv("VLAJEPA_SERVER_HOST", "127.0.0.1")
        server_port = int(os.getenv("VLAJEPA_SERVER_PORT", "10093"))
        self.vlajepa_client = VLAJEPAClient(host=server_host, port=server_port)
        # Action normalization stats announced by the server (dataset_statistics),
        # used to un-normalize [accel, curvature] before the unicycle rollout.
        self._server_norm_stats = self.vlajepa_client.server_norm_stats()
        self._action_space = None  # Alpamayo unicycle action space; built lazily

        self.last_pred_waypoints: torch.Tensor | None = None
        self.last_inference_step = -1
        self.inference_interval = int(os.getenv("VLAJEPA_INFERENCE_INTERVAL", "2"))
        # Training resizes every VLM view to resolution_size x resolution_size (224).
        self.vlajepa_image_size = int(os.getenv("VLAJEPA_IMAGE_SIZE", "224"))

        self.bb_buffer = deque(maxlen=1)
        self.stop_sign_post_processor = StopSignPostProcessor(
            config=self.training_config,
            config_test_time=self.config_closed_loop,
            bb_buffer=self.bb_buffer,
        )
        self.force_move_post_processor = ForceMovePostProcessor(
            config=self.training_config,
            config_test_time=self.config_closed_loop,
            lidar_queue=self.lidar_pc_queue,
        )
        self.metric_info = {}
        self.meters_travelled = 0.0
        self.infraction_recorder = InfractionRecorder(
            config_closed_loop=self.config_closed_loop,
            agent_name="VLAJEPAAgent",
        )
        self.infractions_log = self.infraction_recorder.infractions_log
        self.track = autonomous_agent.Track.SENSORS

        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg is required for LEAD video cleanup/compression.")

    def _build_waypoint_controller(self) -> ClosedLoopInference:
        controller = ClosedLoopInference.__new__(ClosedLoopInference)
        controller.config_expert = self.config_expert
        controller.config_training = self.training_config
        controller.config_closed_loop = self.config_closed_loop
        controller.device = self.device
        controller.lateral_waypoint_controller = PIDController(
            k_p=self.config_closed_loop.turn_kp,
            k_i=self.config_closed_loop.turn_ki,
            k_d=self.config_closed_loop.turn_kd,
            n=self.config_closed_loop.turn_n,
        )
        controller.longitudinal_waypoint_controller = PIDController(
            k_p=self.config_closed_loop.speed_kp,
            k_i=self.config_closed_loop.speed_ki,
            k_d=self.config_closed_loop.speed_kd,
            n=self.config_closed_loop.speed_n,
        )
        return controller

    @beartype
    def set_global_plan(
        self,
        global_plan_gps,
        privileged_org_dense_route_world_coord: list[tuple[carla.Transform, RoadOption]],
    ):
        self.privileged_org_dense_route_world_coord = privileged_org_dense_route_world_coord
        super().set_global_plan(global_plan_gps, privileged_org_dense_route_world_coord)

    def set_scenario(self, scenario):
        self.infraction_recorder.set_scenario(scenario)

    def _init(self):
        self._vehicle: carla.Actor = CarlaDataProvider.get_hero_actor()
        self._world: carla.World = self._vehicle.get_world()
        self.video_recorder = VideoRecorder(
            config_closed_loop=self.config_closed_loop,
            vehicle=self._vehicle,
            world=self._world,
            step_counter=self.step,
            training_config=self.training_config,
        )
        self.initialized = True

    @beartype
    def sensors(self) -> list[dict]:
        # VLA-JEPA drives from the 4 cameras only; LiDAR/radar are unused by the
        # policy and their per-tick simulation just adds latency. Gate them on the
        # same expert-config flags as data collection (LEAD_EXPERT_CONFIG:
        # py123d_disable_lidar / use_radars), matching AlpamayoAgent.
        return av_sensor_setup(
            config=self.training_config,
            lidar=not self.config_expert.py123d_disable_lidar,
            radar=self.config_expert.use_radars,
            sensor_agent=True,
            perturbate=False,
            perturbation_rotation=0.0,
            perturbation_translation=0.0,
        )

    @beartype
    def set_target_points(self, input_data: dict, pop_distance: float):
        planner: RoutePlanner = self.gps_waypoint_planners_dict[pop_distance]

        def transform(point: list[float]) -> np.ndarray:
            ego_position = (
                self.filtered_state[:2]
                if self.config_closed_loop.use_kalman_filter
                else input_data["noisy_state"][:2]
            )
            return common_utils.inverse_conversion_2d(
                np.array(point),
                np.array(ego_position),
                self.compass,
            )

        next_target_points = [tp[0].tolist() for tp in planner.route]
        next_commands = [int(planner.route[i][1]) for i in range(len(planner.route))]

        filtered_tp_list = []
        filtered_command_list = []
        for pt, cmd in zip(next_target_points, next_commands, strict=False):
            if (
                len(next_target_points) == 2
                or not filtered_tp_list
                or not np.allclose(pt[:2], filtered_tp_list[-1][:2])
            ):
                filtered_tp_list.append(pt)
                filtered_command_list.append(cmd)

        next_target_points = filtered_tp_list
        next_commands = filtered_command_list
        if len(next_target_points) > 2:
            input_data["target_point_next"] = transform(next_target_points[2][:2])
            input_data["target_point"] = transform(next_target_points[1][:2])
            input_data["target_point_previous"] = transform(next_target_points[0][:2])
        else:
            input_data["target_point_next"] = transform(next_target_points[1][:2])
            input_data["target_point"] = transform(next_target_points[1][:2])
            input_data["target_point_previous"] = transform(next_target_points[0][:2])

        input_data["route_command"] = next_commands[0]

    @beartype
    @torch.inference_mode()
    def tick(self, input_data: dict) -> dict:
        input_data = super().tick(
            input_data,
            use_kalman_filter=self.training_config.use_kalman_filter_for_gps,
        )
        self.set_target_points(
            input_data,
            pop_distance=self.config_closed_loop.route_planner_min_distance,
        )
        if self.config_closed_loop.sensor_agent_pop_distance_adaptive:
            dense_points = (
                np.linalg.norm(input_data["target_point"] - input_data["target_point_next"]) < 10.0
                and min(
                    np.linalg.norm(input_data["target_point_previous"]),
                    np.linalg.norm(input_data["target_point"]),
                )
                < 10.0
            )
            if dense_points:
                self.set_target_points(input_data, pop_distance=4.0)

        return input_data

    def _vlajepa_images(self, input_data: dict) -> list[np.ndarray]:
        """Return the 4 camera frames as RGB arrays in the training video-key order.

        Training video_keys order is [front_wide, cross_left, front_tele, cross_right].
        LEAD's rig maps rgb_1..4 -> [cross_left, front_wide, cross_right, front_tele],
        so the requested order is [rgb_2, rgb_1, rgb_4, rgb_3].

        Each view is square-resized to ``resolution_size`` (224) to match training:
        the dataloader feeds ``Image.fromarray(video[0]).resize((224, 224))`` to the
        VLM. predict_action's own resize is gated on a non-existent ``image_size``
        config key, so it never fires; we resize here so the VLM sees the same
        resolution/aspect it was trained on.
        """
        order = [2, 1, 4, 3]
        size = self.vlajepa_image_size
        images = []
        for camera_idx in order:
            bgr = input_data[f"rgb_{camera_idx}"]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
            images.append(rgb)
        return images

    def _navigation_command_text(self, route_command: int) -> str:
        command_map = {
            1: "Turn left",
            2: "Turn right",
            3: "Go straight",
            4: "Follow the lane",
            5: "Change lane left",
            6: "Change lane right",
        }
        return command_map.get(int(route_command), "Follow the lane")

    def _build_action_space(self, n_waypoints: int):
        """Build Alpamayo 1.5's unicycle action space seeded with the server's
        accel/curvature normalization stats (action_to_traj un-normalizes with them)."""
        import sys
        from pathlib import Path

        stats = self._server_norm_stats
        if not stats:
            raise RuntimeError(
                "Server announced no norm_stats; cannot un-normalize [accel, curvature]. "
                "Ensure the policy server run dir has dataset_statistics.json."
            )
        action_stats = stats[next(iter(stats))]["action"]
        mean = np.asarray(action_stats["mean"], dtype=np.float64)
        std = np.asarray(action_stats["std"], dtype=np.float64)

        alpamayo15 = Path(__file__).resolve().parents[2] / "3rd_party" / "alpamayo1.5" / "src"
        if str(alpamayo15) not in sys.path:
            sys.path.insert(0, str(alpamayo15))
        from alpamayo1_5.action_space.unicycle_accel_curvature import (
            UnicycleAccelCurvatureActionSpace,
        )

        return UnicycleAccelCurvatureActionSpace(
            accel_mean=float(mean[0]),
            accel_std=float(std[0]),
            curvature_mean=float(mean[1]),
            curvature_std=float(std[1]),
            dt=1.0 / self.training_config.carla_fps * self.training_config.data_save_freq,
            n_waypoints=n_waypoints,
        )

    def _rollout_actions(self, actions_norm: np.ndarray, ego_speed: float) -> np.ndarray:
        """Roll out normalized unicycle controls [N,2] (accel, curvature) into
        ego-frame waypoints [N,2] via Alpamayo's ``action_to_traj``, seeded with the
        current speed. The action space un-normalizes internally. The resulting
        frame is ISO 8855 (x forward, y LEFT), matching LEAD's PID convention.
        """
        if self._action_space is None or self._action_space.n_waypoints != actions_norm.shape[0]:
            self._action_space = self._build_action_space(actions_norm.shape[0])
        act = torch.from_numpy(actions_norm.astype(np.float64)).unsqueeze(0)  # (1,N,2)
        hist = torch.zeros(1, 2, 3, dtype=torch.float64)
        hist_rot = torch.eye(3, dtype=torch.float64)[None, None].repeat(1, 2, 1, 1)
        future_xyz, _ = self._action_space.action_to_traj(
            act,
            hist,
            hist_rot,
            t0_states={"v": torch.tensor([float(ego_speed)], dtype=torch.float64)},
        )
        return future_xyz[0, :, :2].numpy().astype(np.float32)

    def check_infractions(self) -> None:
        self.infraction_recorder.check_infractions(
            step=self.step,
            meters_travelled=self.meters_travelled,
        )

    @beartype
    @torch.inference_mode()
    def run_step(self, input_data: dict, _, __=None) -> carla.VehicleControl:
        self.step += 1

        if not self.initialized:
            self._init()
            self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.tick(input_data)
            return self.control

        if hasattr(self, "video_recorder"):
            self.video_recorder.update_step(self.step)
            self.video_recorder.move_demo_cameras_with_ego()

        input_data = self.tick(input_data)
        should_infer = (
            self.last_pred_waypoints is None
            or self.step - self.last_inference_step >= self.inference_interval
        )
        if should_infer:
            images = self._vlajepa_images(input_data)
            instruction = self._navigation_command_text(input_data.get("route_command", 4))
            actions = self.vlajepa_client.predict_actions(images, instruction)
            waypoints = self._rollout_actions(actions, float(input_data["speed"]))
            self.last_pred_waypoints = torch.from_numpy(waypoints).unsqueeze(0).to(self.device)
            self.last_inference_step = self.step

        ego_speed = torch.tensor([[input_data["speed"]]], device=self.device, dtype=torch.float32)
        steer, throttle, brake = self.closed_loop_controller.execute_waypoints(
            self.last_pred_waypoints.to(self.device),
            ego_speed,
        )

        throttle, brake = self.force_move_post_processor.adjust(
            float(input_data["speed"]),
            throttle,
            brake,
        )
        throttle, brake = self.stop_sign_post_processor.adjust(
            float(input_data["speed"]),
            throttle,
            brake,
        )
        self.meters_travelled += float(input_data["speed"]) * self.config_closed_loop.carla_frame_rate
        input_data["meters_travelled"] = self.meters_travelled

        self.control = carla.VehicleControl(
            steer=float(steer),
            throttle=float(throttle),
            brake=float(brake),
        )
        if self.step < self.training_config.inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)

        self.check_infractions()
        self._save_video_frames(input_data)

        if self.config_closed_loop.is_bench2drive and hasattr(self, "get_metric_info"):
            self.metric_info[self.step] = self.get_metric_info()
            with open(
                f"{self.config_closed_loop.save_path}/metric_info.json",
                "w",
                encoding="utf-8",
            ) as outfile:
                json.dump(self.metric_info, outfile, indent=4)
        return self.control

    def _save_video_frames(self, input_data: dict) -> None:
        if not hasattr(self, "video_recorder"):
            return
        if self.config_closed_loop.save_path is None:
            return
        if self.step % self.config_closed_loop.produce_frame_frequency != 0:
            return

        input_image = input_data["rgb"].copy()
        if self.config_closed_loop.produce_input_image:
            self.video_recorder.save_input_image(input_image)
        if self.config_closed_loop.produce_input_video:
            self.video_recorder.save_input_video_frame(input_image)

    def destroy(self, results=None):
        LOG.info(results)
        if hasattr(self, "vlajepa_client"):
            self.vlajepa_client.close()
        if hasattr(self, "video_recorder"):
            self.video_recorder.cleanup_and_compress()


class StopSignPostProcessor:
    """Placeholder hook matching SensorAgent's stop-sign postprocessor API."""

    def __init__(self, config: TrainingConfig, config_test_time: ClosedLoopConfig, bb_buffer: deque):
        self.config = config
        self.config_test_time = config_test_time
        self.bb_buffer = bb_buffer

    def adjust(self, ego_speed: float, current_throttle: float, current_brake: float):
        return current_throttle, current_brake


class ForceMovePostProcessor:
    """Forces the agent to creep after a long standstill, using LEAD's safety box."""

    def __init__(
        self,
        config: TrainingConfig,
        config_test_time: ClosedLoopConfig,
        lidar_queue: deque,
    ):
        self.config = config
        self.config_test_time = config_test_time
        self.stuck_detector = 0
        self.force_move = 0
        self.lidar_buffer = lidar_queue

    def adjust(
        self,
        ego_speed: float,
        current_throttle: float,
        current_brake: float,
    ) -> tuple[float, float]:
        if not self.config_test_time.sensor_agent_creeping:
            return current_throttle, current_brake
        if ego_speed < 0.1:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0

        if self.stuck_detector > self.config_test_time.sensor_agent_stuck_threshold:
            self.force_move = self.config_test_time.sensor_agent_stuck_move_duration

        if self.force_move <= 0:
            return current_throttle, current_brake

        emergency_stop = False
        if len(self.lidar_buffer) > 0:
            safety_box = deepcopy(self.lidar_buffer[-1])
            safety_box = safety_box[safety_box[..., 2] > self.config.safety_box_z_min]
            safety_box = safety_box[safety_box[..., 2] < self.config.safety_box_z_max]
            safety_box = safety_box[safety_box[..., 1] > self.config.safety_box_y_min]
            safety_box = safety_box[safety_box[..., 1] < self.config.safety_box_y_max]
            safety_box = safety_box[safety_box[..., 0] > self.config.safety_box_x_min]
            safety_box = safety_box[safety_box[..., 0] < self.config.safety_box_x_max]
            emergency_stop = len(safety_box) > 0

        if emergency_stop:
            self.force_move = self.config_test_time.sensor_agent_stuck_move_duration
            return 0.0, 1.0

        self.force_move -= 1
        return max(self.config_test_time.sensor_agent_stuck_throttle, current_throttle), 0.0


if __name__ == "__main__":
    agent = VLAJEPAAgent()
