from __future__ import annotations

import numpy as np
from py123d.datatypes import CameraID, FThetaCameraMetadata
from py123d.datatypes.sensors.ftheta_camera import FThetaIntrinsics
from py123d.geometry import PoseSE3
from py123d.geometry.utils.rotation_utils import get_quaternion_array_from_rotation_matrix

from py123d_garage.config.schema.evaluation.alpasim_config import AebConfig
from py123d_garage.evaluation.alpasim.help import aeb


def _front_camera() -> FThetaCameraMetadata:
    """An equidistant fisheye 1.5 m up at the rig origin, looking along the rig's x."""
    focal = 500.0
    # Camera x right, y down, z forward, as columns in the rig frame (x forward, y left, z up).
    rotation = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    qw, qx, qy, qz = get_quaternion_array_from_rotation_matrix(rotation)
    return FThetaCameraMetadata(
        camera_name="camera_front_wide_120fov",
        camera_id=CameraID.FTCAM_F0,
        intrinsics=FThetaIntrinsics(
            cx=960.0,
            cy=540.0,
            fw_poly=np.array([0.0, focal, 0.0, 0.0, 0.0, 0.0]),
            bw_poly=np.array([0.0, 1.0 / focal, 0.0, 0.0, 0.0, 0.0]),
        ),
        width=1920,
        height=1080,
        camera_to_imu_se3=PoseSE3(x=0.0, y=0.0, z=1.5, qw=qw, qx=qx, qy=qy, qz=qz),
    )


def test_box_bottoms_land_on_the_ground_where_the_road_user_stands() -> None:
    camera = _front_camera()
    rotation = camera.camera_to_imu_se3.rotation_matrix
    ground_rig = np.array([[12.0, 0.8, 0.0]])
    pixels, _, _ = camera.project_to_image((ground_rig - [0.0, 0.0, 1.5]) @ rotation)
    u, v = pixels[0]
    (points,) = aeb.ground_points_rig(np.array([[u, v - 40.0, u, v]]), camera)
    np.testing.assert_allclose(points, [[12.0, 0.8]] * len(points), atol=1e-6)
    # A box ending above the horizon casts no ground point.
    (sky,) = aeb.ground_points_rig(np.array([[900.0, 100.0, 1000.0, 300.0]]), camera)
    assert len(sky) == 0


def test_lead_distance_counts_only_road_users_inside_the_corridor() -> None:
    config = AebConfig()
    path = np.stack([np.arange(1.0, 30.0), np.zeros(29)], axis=-1)
    ahead = [np.array([[20.0, 0.5], [20.0, 2.5]])]
    assert np.isclose(aeb.lead_distance_m(ahead, path, config), 20.0 - config.front_overhang_m)
    assert aeb.lead_distance_m([np.array([[20.0, 3.0]])], path, config) is None
    # Beyond the planned path the corridor runs straight on along its last heading.
    assert np.isclose(aeb.lead_distance_m([np.array([[50.0, 0.0]])], path, config), 50.0 - config.front_overhang_m)


def test_speed_cap_brakes_for_a_fast_approach_and_holds_speed_for_a_slow_one() -> None:
    config = AebConfig()
    assert aeb.speed_cap(None, None, 10.0, config) is None
    assert aeb.speed_cap(20.0, -1.0, 10.0, config) is None
    decel, floor = aeb.speed_cap(15.0, 10.0, 10.0, config)
    assert decel >= 3.0
    assert floor == 0.0
    # 12 m at 5 m/s closing: no hurry to brake, but no speeding up either.
    assert aeb.speed_cap(12.0, 5.0, 12.0, config) == (0.0, 12.0)
    assert aeb.speed_cap(60.0, 5.0, 12.0, config) is None


def test_lead_track_fits_the_closing_speed() -> None:
    config = AebConfig()
    track = aeb.LeadTrack()
    for step in range(config.min_track_samples):
        track.update(step * 0.1, 30.0 - 0.8 * step, config)
    assert np.isclose(track.closing_speed_mps(config), 8.0)
    # A jump to another road user restarts the track.
    track.update(1.0, 10.0, config)
    assert track.closing_speed_mps(config) is None


def test_retimed_plan_brakes_along_the_path_from_where_the_ego_stands() -> None:
    times_s = np.arange(0.0, 5.01, 0.5)
    positions_xy = np.stack([10.0 * times_s, np.zeros_like(times_s)], axis=-1)
    yaws = np.zeros_like(times_s)
    taus, xy, _ = aeb.retime_plan(times_s, positions_xy, yaws, 0.0, np.zeros(2), 10.0, 5.0, 0.0)
    assert np.isclose(taus[1] - taus[0], 0.1)
    # 10 m/s braking at 5 m/s^2 stops after 10 m and stays there.
    np.testing.assert_allclose(xy[taus >= 2.0 - 1e-9, 0], 10.0)
    np.testing.assert_allclose(xy[10, 0], 10.0 - 0.5 * 5.0)
    # One second in, the ego lags 3 m behind the plan: it starts where it stands, not where the plan would be.
    taus, xy, _ = aeb.retime_plan(times_s, positions_xy, yaws, 1.0, np.array([7.0, 0.2]), 10.0, 0.0, 10.0)
    np.testing.assert_allclose(xy[0], [7.0, 0.0])
    np.testing.assert_allclose(xy[10, 0], 17.0)
