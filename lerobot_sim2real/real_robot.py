from pathlib import Path

from lerobot.robots.robot import Robot
from lerobot.robots.so_follower.config_so_follower import SO100FollowerConfig, SO101FollowerConfig
from lerobot.robots.utils import make_robot_from_config

# so101-lab calibrate_leader.py writes <name>.json here. MoveIt drives the follower only.
FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A054769-if00"
FOLLOWER_CALIBRATION_DIR = Path(
    "/home/cym/ROS/Arms/isaac_so_arm101/so101-lab/so101_lab/devices/lerobot/.cache"
)


def create_real_robot(uid: str = "so100") -> Robot:
    """Wrapper function to map string UIDS to real robot configurations. Primarily for saving a bit of code for users when they fork the repository. They can just edit the camera, id etc. settings in this one file."""
    if uid == "so100":
        robot_config = SO100FollowerConfig(
            port="COM24",
            id="so100_follower",
            use_degrees=True,
            # for phone camera users you can use the commented out setting below
            # cameras={
            #     "base_camera": OpenCVCameraConfig(camera_index=1, fps=30, width=640, height=480)
            # }
            # for intel realsense camera users you need to modify the serial number or name for your own hardware
            # cameras={
            #     "base_camera": OpenCVCameraConfig(
            #         index_or_path=Path("/dev/video2"),
            #         height=1080,
            #         width=1920,
            #         fps=30,
            #         warmup_s=2,
            #         )
            # },
        )
    elif uid == "so101":
        robot_config = SO101FollowerConfig(
            # Follower serial 5A7A054769. Leader is 5A7A057348 / leader_1.json.
            port=FOLLOWER_PORT,
            id="follower_1",
            calibration_dir=FOLLOWER_CALIBRATION_DIR,
            use_degrees=True,
            max_relative_target=10.0,  # degrees per step, safety limit against sudden jumps
            # for phone camera users you can use the commented out setting below
            # cameras={
            #     "base_camera": OpenCVCameraConfig(camera_index=1, fps=30, width=640, height=480)
            # }
            # for intel realsense camera users you need to modify the serial number or name for your own hardware
            # cameras={
            #     "base_camera": OpenCVCameraConfig(
            #         index_or_path=2,
            #         height=1080,
            #         width=1920,
            #         fps=30,
            #         warmup_s=2,
            #         )
            # },
        )
    else:
        raise ValueError(f"Invalid robot UID: {uid}")
    real_robot = make_robot_from_config(robot_config)
    return real_robot
