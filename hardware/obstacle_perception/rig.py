from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any

import numpy as np

TOP_CAMERA_SERIAL = "323622271046"
LEFT_CAMERA_SERIAL = "335122272969"
MODEL_ID = "facebook/sam2.1-hiera-small"


def transform_from_rpy(rpy: np.ndarray, translation: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation_x = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        dtype=np.float64,
    )
    rotation_y = np.asarray(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
        dtype=np.float64,
    )
    rotation_z = np.asarray(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_z @ rotation_y @ rotation_x
    transform[:3, 3] = translation
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if not len(points):
        return points.copy()
    return points @ transform[:3, :3].T + transform[:3, 3]


class PiperPoseReader:
    def __init__(self, can_interface: str):
        self.can_interface = can_interface
        self.interface: Any = None

    def connect(self) -> None:
        can_path = Path("/sys/class/net") / self.can_interface
        if (
            not can_path.exists()
            or (can_path / "operstate").read_text().strip() != "up"
        ):
            raise RuntimeError(
                f"CAN interface {self.can_interface} is not up; "
                "activate it with piper_sdk's can_activate.sh"
            )
        module = importlib.import_module("piper_sdk")
        self.interface = module.C_PiperInterface_V2(self.can_interface)
        self.interface.ConnectPort(piper_init=False)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self.read() is not None:
                return
            time.sleep(0.05)
        self.disconnect()
        raise RuntimeError(f"No end-pose feedback on {self.can_interface}")

    def read(self) -> np.ndarray | None:
        if self.interface is None:
            return None
        message = self.interface.GetArmEndPoseMsgs()
        if float(message.time_stamp) <= 0.0 or float(message.Hz) <= 0.0:
            return None
        pose = message.end_pose
        translation = (
            np.asarray([pose.X_axis, pose.Y_axis, pose.Z_axis], dtype=np.float64)
            / 1_000_000.0
        )
        rpy = np.radians(
            np.asarray([pose.RX_axis, pose.RY_axis, pose.RZ_axis], dtype=np.float64)
            / 1000.0
        )
        return transform_from_rpy(rpy, translation)

    def read_state(self) -> tuple[float, ...] | None:
        if self.interface is None:
            return None
        joint_message = self.interface.GetArmJointMsgs()
        gripper_message = self.interface.GetArmGripperMsgs()
        if (
            float(joint_message.time_stamp) <= 0.0
            or float(gripper_message.time_stamp) <= 0.0
        ):
            return None
        joints = joint_message.joint_state
        state = [
            float(getattr(joints, f"joint_{index}")) / 1000.0 for index in range(1, 7)
        ]
        state.append(float(gripper_message.gripper_state.grippers_angle) / 10000.0)
        return tuple(state)

    def disconnect(self) -> None:
        if self.interface is not None:
            self.interface.DisconnectPort()
            self.interface = None
