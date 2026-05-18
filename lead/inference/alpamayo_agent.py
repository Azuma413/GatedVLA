import json
import logging
import os
import shutil
from collections import deque

import carla
import cv2
import numpy as np
import torch
from agents.navigation.local_planner import RoadOption
from beartype import beartype
from leaderboard.autoagents import autonomous_agent
from scipy.spatial.transform import Rotation
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from lead.common import common_utils
from lead.common.base_agent import BaseAgent
from lead.common.constants import TargetDataset
from lead.common.pid_controller import PIDController
from lead.common.route_planner import RoutePlanner
from lead.common.sensor_setup import av_sensor_setup
from lead.inference.alpamayo_inference import AlpamayoInference
from lead.inference.closed_loop_inference import ClosedLoopInference
from lead.inference.config_closed_loop import ClosedLoopConfig
from lead.inference.infraction_recorder import InfractionRecorder
from lead.inference.sensor_agent import ForceMovePostProcessor, StopSignPostProcessor
from lead.inference.video_recorder import VideoRecorder
from lead.training.config_training import TrainingConfig

LOG = logging.getLogger(__name__)


def get_entry_point():
    return "AlpamayoAgent"


class AlpamayoEvalConfig(TrainingConfig):
    """LEAD sensor config that fixes the online rig to Alpamayo's 4 cameras."""

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


class AlpamayoAgent(BaseAgent, autonomous_agent.AutonomousAgent):
    @beartype
    def setup(self, path_to_conf_file: str, _=None, __=None):
        self.config_closed_loop = ClosedLoopConfig()
        super().setup(sensor_agent=True)
        self.config_path = path_to_conf_file.split("+")[0]
        self.step = -1
        self.initialized = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.training_config = AlpamayoEvalConfig({"target_dataset": 7})
        self.closed_loop_controller = self._build_waypoint_controller()

        self.alpamayo_inference = AlpamayoInference(
            checkpoint_path=self.config_path,
            device=self.device,
        )

        self.camera_frame_history: deque[dict[int, np.ndarray]] = deque(maxlen=4)
        self.last_pred_waypoints: torch.Tensor | None = None
        self.last_inference_step = -1
        self.inference_interval = int(os.getenv("ALPAMAYO_INFERENCE_INTERVAL", "2"))

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
            agent_name="AlpamayoAgent",
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
        return av_sensor_setup(
            config=self.training_config,
            lidar=True,
            radar=True,
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

    def _current_camera_frames(self, input_data: dict) -> dict[int, np.ndarray]:
        frames = {}
        for camera_idx in range(1, 5):
            bgr = input_data[f"rgb_{camera_idx}"]
            frames[camera_idx] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return frames

    def _ego_history_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        positions = np.asarray(self.ego_past_positions, dtype=np.float32)
        yaws = np.asarray(self.ego_past_yaws, dtype=np.float32)
        if positions.shape[0] == 0:
            positions = np.zeros((1, 2), dtype=np.float32)
            yaws = np.zeros((1,), dtype=np.float32)

        if positions.shape[0] < 16:
            pad_count = 16 - positions.shape[0]
            positions = np.concatenate([np.repeat(positions[:1], pad_count, axis=0), positions])
            yaws = np.concatenate([np.repeat(yaws[:1], pad_count, axis=0), yaws])
        else:
            positions = positions[-16:]
            yaws = yaws[-16:]

        xyz = np.zeros((1, 16, 3), dtype=np.float32)
        xyz[0, :, :2] = positions
        rot = Rotation.from_euler("z", yaws).as_matrix().astype(np.float32)[None]
        return torch.from_numpy(xyz), torch.from_numpy(rot)

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

    def _build_alpamayo_sample(self, input_data: dict) -> dict:
        current_frames = self._current_camera_frames(input_data)
        self.camera_frame_history.append(current_frames)
        while len(self.camera_frame_history) < 4:
            self.camera_frame_history.appendleft(current_frames)

        camera_tensors = []
        for camera_idx in range(1, 5):
            frame_stack = [frames[camera_idx] for frames in self.camera_frame_history]
            frames_np = np.stack(frame_stack, axis=0)
            frames_np = np.transpose(frames_np, (0, 3, 1, 2))
            camera_tensors.append(torch.from_numpy(frames_np.copy()).to(torch.uint8))

        ego_history_xyz, ego_history_rot = self._ego_history_tensors()
        relative_timestamps = torch.tensor(
            [[0.0, 0.1, 0.2, 0.3]] * 4,
            dtype=torch.float32,
        )
        return {
            "image_frames": torch.stack(camera_tensors, dim=0),
            "camera_indices": torch.tensor([0, 1, 2, 6], dtype=torch.int64),
            "relative_timestamps": relative_timestamps,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "navigation_command_text": self._navigation_command_text(
                input_data.get("route_command", 4),
            ),
        }

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
            sample_data = self._build_alpamayo_sample(input_data)
            self.last_pred_waypoints = self.alpamayo_inference.predict_waypoints(sample_data)
            self.last_inference_step = self.step
        else:
            self._build_alpamayo_sample(input_data)

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

        if self.config_closed_loop.is_bench2drive and hasattr(self, "get_metric_info"):
            self.metric_info[self.step] = self.get_metric_info()
            with open(
                f"{self.config_closed_loop.save_path}/metric_info.json",
                "w",
                encoding="utf-8",
            ) as outfile:
                json.dump(self.metric_info, outfile, indent=4)
        return self.control

    def destroy(self, results=None):
        LOG.info(results)
        if hasattr(self, "video_recorder"):
            self.video_recorder.cleanup_and_compress()


if __name__ == "__main__":
    agent = AlpamayoAgent()
