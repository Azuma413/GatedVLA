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
from scipy.spatial.transform import Rotation
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from lead.common import common_utils
from lead.common.base_agent import BaseAgent
from lead.common.constants import TargetDataset
from lead.common.pid_controller import ExpertLongitudinalController, PIDController
from lead.common.route_planner import RoutePlanner
from lead.common.sensor_setup import av_sensor_setup
from lead.inference.alpamayo_inference import AlpamayoInference
from lead.inference.closed_loop_inference import ClosedLoopInference
from lead.inference.config_closed_loop import ClosedLoopConfig
from lead.inference.infraction_recorder import InfractionRecorder
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

        # Alpamayo SFT uses 16 history steps + 4 image frames at 0.1s spacing (matches the
        # PAI export's 10fps videos and load_physical_aiavdataset time_step=0.1). CARLA ticks
        # at carla_fps, so subsample every `frame_stride` ticks to hit 0.1s.
        self.alpamayo_time_step = 0.1
        self.num_history_steps = 16
        self.num_image_frames = 4
        self.frame_stride = max(1, round(self.alpamayo_time_step * self.training_config.carla_fps))
        # Diagnostic toggle: whether to convert LEAD/CARLA ego frame <-> ISO 8855 (SFT export
        # frame) by flipping y / yaw. Default off keeps the original (pre-fix) coordinate handling.
        self.iso_frame_flip = os.getenv("ALPAMAYO_ISO_FRAME", "0") == "1"
        # Diagnostic: log nav_text / route_command / raw+used waypoints / steer per inference.
        self.debug_nav = os.getenv("ALPAMAYO_DEBUG_NAV", "0") == "1"
        # Diagnostic: optionally enable the anti-stuck creep (disabled by default in
        # ClosedLoopConfig) to test recovery from the zero-velocity deadlock.
        if os.getenv("ALPAMAYO_CREEP", "0") == "1":
            self.config_closed_loop.sensor_agent_creeping = True
            self.config_closed_loop.sensor_agent_stuck_threshold = int(
                os.getenv("ALPAMAYO_STUCK_THRESHOLD", "40"),
            )
        # Match data-collection longitudinal control: replace execute_waypoints' bang-bang
        # throttle/brake with the expert's tuned ExpertLongitudinalController (the controller
        # that produced the SFT speed profiles). Steering stays pure-pursuit (already turns
        # correctly; the expert lateral PID assumes a different waypoint spacing). Target speed
        # is derived from the model trajectory's near-term implied speed.
        self.use_expert_long = os.getenv("ALPAMAYO_EXPERT_LONG", "0") == "1"
        self.expert_long_controller = ExpertLongitudinalController(self.config_expert)
        self.camera_frame_history: deque[dict[int, np.ndarray]] = deque(
            maxlen=(self.num_image_frames - 1) * self.frame_stride + 1,
        )
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

    def _current_camera_frames(self, input_data: dict) -> dict[int, np.ndarray]:
        frames = {}
        for camera_idx in range(1, 5):
            bgr = input_data[f"rgb_{camera_idx}"]
            frames[camera_idx] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return frames

    def _ego_history_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        # ego_past_positions/yaws are per-tick (carla_fps), oldest -> current, in the LEAD/CARLA
        # ego frame (x forward, y right). Alpamayo trains on num_history_steps at 0.1s spacing in
        # the ISO 8855 ego frame (x forward, y left, yaw CCW-positive), so subsample every
        # frame_stride ticks and flip y / yaw to match the SFT export.
        positions = np.asarray(self.ego_past_positions, dtype=np.float32)
        yaws = np.asarray(self.ego_past_yaws, dtype=np.float32)
        if positions.shape[0] == 0:
            positions = np.zeros((1, 2), dtype=np.float32)
            yaws = np.zeros((1,), dtype=np.float32)

        # newest-first every frame_stride ticks, keep num_history_steps, back to oldest -> current
        positions = positions[::-1][:: self.frame_stride][: self.num_history_steps][::-1]
        yaws = yaws[::-1][:: self.frame_stride][: self.num_history_steps][::-1]

        if positions.shape[0] < self.num_history_steps:
            pad_count = self.num_history_steps - positions.shape[0]
            positions = np.concatenate([np.repeat(positions[:1], pad_count, axis=0), positions])
            yaws = np.concatenate([np.repeat(yaws[:1], pad_count, axis=0), yaws])

        # CARLA (y right, yaw CW) -> ISO 8855 (y left, yaw CCW), matching the SFT export.
        if self.iso_frame_flip:
            positions = positions.copy()
            positions[:, 1] *= -1.0
            yaws = -yaws

        xyz = np.zeros((1, self.num_history_steps, 3), dtype=np.float32)
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
        # Alpamayo trains on num_image_frames at 0.1s spacing (t0-0.3..t0). The buffer fills per
        # tick, so subsample every frame_stride ticks (newest first), order oldest -> current,
        # and pad by repeating the oldest during warm-up.
        selected = list(self.camera_frame_history)[::-1][:: self.frame_stride][
            : self.num_image_frames
        ][::-1]
        while len(selected) < self.num_image_frames:
            selected.insert(0, selected[0])

        camera_tensors = []
        for camera_idx in range(1, 5):
            frame_stack = [frames[camera_idx] for frames in selected]
            frames_np = np.stack(frame_stack, axis=0)
            frames_np = np.transpose(frames_np, (0, 3, 1, 2))
            camera_tensors.append(torch.from_numpy(frames_np.copy()).to(torch.uint8))

        ego_history_xyz, ego_history_rot = self._ego_history_tensors()
        rel_ts = [round(i * self.alpamayo_time_step, 3) for i in range(self.num_image_frames)]
        relative_timestamps = torch.tensor([rel_ts] * 4, dtype=torch.float32)
        return {
            "image_frames": torch.stack(camera_tensors, dim=0),
            "camera_indices": torch.tensor([0, 1, 2, 6], dtype=torch.int64),
            "relative_timestamps": relative_timestamps,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "navigation_command_text": self._navigation_command_text(
                input_data.get("route_command", 4),
            ),
            # r1_5 route conditioning reads data["nav_text"]; matches the PAI
            # navigation_command command_text used during 1.5 SFT.
            "nav_text": self._navigation_command_text(
                input_data.get("route_command", 4),
            ),
        }

    def _trajectory_target_speed(self) -> float:
        # Near-term implied speed of the predicted trajectory: median per-step distance over the
        # first ~1s (robust to noise) divided by the 0.1s step. Frame-independent (uses norms).
        wp = self.last_pred_waypoints[0].detach().cpu().numpy()
        n = min(11, wp.shape[0])
        if n < 2:
            return 0.0
        steps = np.linalg.norm(np.diff(wp[:n], axis=0), axis=1)
        return float(np.median(steps)) / self.alpamayo_time_step

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
            raw_pred = self.alpamayo_inference.predict_waypoints(sample_data)
            # Model predicts in the ISO 8855 ego frame (y left); the LEAD controller expects the
            # CARLA ego frame (y right). Flip y back before control.
            if self.iso_frame_flip:
                pred_waypoints = raw_pred.clone()
                pred_waypoints[..., 1] *= -1.0
            else:
                pred_waypoints = raw_pred
            self.last_pred_waypoints = pred_waypoints
            self.last_inference_step = self.step
            if self.debug_nav:
                self._dbg = {
                    "route_command": input_data.get("route_command"),
                    "nav_text": sample_data["nav_text"],
                    "raw_wp": raw_pred[0, :6, :].tolist(),
                    "used_wp": pred_waypoints[0, :6, :].tolist(),
                }
        else:
            self._build_alpamayo_sample(input_data)

        ego_speed = torch.tensor([[input_data["speed"]]], device=self.device, dtype=torch.float32)
        steer, throttle, brake = self.closed_loop_controller.execute_waypoints(
            self.last_pred_waypoints.to(self.device),
            ego_speed,
        )

        model_throttle, model_brake = float(throttle), float(brake)

        if self.use_expert_long:
            # Keep pure-pursuit steer; replace throttle/brake with the expert's tuned
            # longitudinal controller fed the model trajectory's implied target speed.
            target_speed = self._trajectory_target_speed()
            throttle_e, brake_bool = self.expert_long_controller.get_throttle_and_brake(
                hazard_brake=False,
                target_speed=target_speed,
                current_speed=float(input_data["speed"]),
            )
            throttle, brake = float(throttle_e), (1.0 if brake_bool else 0.0)

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

        if self.debug_nav and should_infer:
            loc = self._vehicle.get_transform().location
            fm = self.force_move_post_processor
            LOG.info(
                "NAVDBG step=%d pos=(%.1f,%.1f) yaw=%.1f spd=%.2f cmd=%s nav=%r steer=%.3f "
                "model(thr=%.2f,brk=%.2f) final(thr=%.2f,brk=%.2f) stuck=%d force_move=%d used_wp=%s",
                self.step, loc.x, loc.y,
                self._vehicle.get_transform().rotation.yaw,
                float(input_data["speed"]),
                self._dbg["route_command"], self._dbg["nav_text"], float(steer),
                model_throttle, model_brake, float(throttle), float(brake),
                fm.stuck_detector, fm.force_move,
                [[round(c, 2) for c in p] for p in self._dbg["used_wp"]],
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
    agent = AlpamayoAgent()
