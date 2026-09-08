from __future__ import annotations

import importlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from .rig import CALIBRATION_DIR, LEFT_CAMERA_SERIAL, TOP_CAMERA_SERIAL

OUTPUT_PATH = CALIBRATION_DIR / "top_from_left.json"
DICTIONARY = "DICT_APRILTAG_16h5"
MARKER_MM = 35.0
BOARD_TAGS = {
    0: (0.00, 0.00, 0.00),
    1: (-57.70, 0.04, 30.06),
    2: (-115.46, -0.07, 90.03),
    3: (-0.03, 52.93, 44.98),
    4: (-57.65, 52.95, -0.01),
    5: (-115.35, 52.96, 59.95),
    6: (-0.04, 108.23, 89.98),
    7: (-57.65, 108.29, 45.00),
    8: (-115.37, 108.32, -0.04),
    9: (0.05, 166.22, -0.05),
    10: (-57.70, 166.26, 29.97),
    11: (-115.51, 166.46, 44.97),
}
WIDTH = 1280
HEIGHT = 720
FPS = 15
POSES = 12
MIN_POSES = 6
MIN_TAGS = 4
MAX_REPROJECTION_PX = 3.0
MIN_TRANSLATION_M = 0.015
MIN_ROTATION_DEG = 4.0
STABILITY_S = 0.35
MAX_STABILITY_TRANSLATION_M = 0.015
MAX_STABILITY_ROTATION_DEG = 3.0
MAX_TRANSLATION_RESIDUAL_M = 0.02
MAX_ROTATION_RESIDUAL_DEG = 2.5


@dataclass
class BoardObservation:
    transform: np.ndarray
    corners: list[np.ndarray]
    ids: np.ndarray
    reprojection_error: float


@dataclass
class PosePair:
    top: BoardObservation
    left: BoardObservation


@dataclass
class CalibrationResult:
    transform: np.ndarray
    translation_errors_m: np.ndarray
    rotation_errors_deg: np.ndarray
    inliers: np.ndarray


def dictionary_from_name(name: str):
    dictionary_id = getattr(cv2.aruco, name, None)
    if dictionary_id is None or not name.startswith("DICT_"):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def board_from_placements(
    placements: dict[int, tuple[float, float, float]], marker_mm: float, dictionary
):
    half = marker_mm / 2000.0
    square = np.asarray([[-half, half], [half, half], [half, -half], [-half, -half]])
    corners = []
    for x_mm, y_mm, angle_deg in placements.values():
        angle = np.radians(angle_deg)
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        planar = square @ rotation.T + np.asarray([x_mm, y_mm]) / 1000.0
        corners.append(
            np.concatenate([planar, np.zeros((4, 1))], axis=1).astype(np.float32)
        )
    return cv2.aruco.Board(
        corners, dictionary, np.asarray(list(placements), dtype=np.int32)
    )


def intrinsics_values(intrinsics) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return matrix, np.asarray(intrinsics.coeffs, dtype=np.float64)


def detector_from_dictionary(dictionary):
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.minMarkerPerimeterRate = 0.01
    parameters.minCornerDistanceRate = 0.01
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def transform_from_pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = cv2.Rodrigues(rotation)[0]
    transform[:3, 3] = translation.reshape(3)
    return transform


def observe_board(
    image: np.ndarray,
    board,
    detector,
    matrix: np.ndarray,
    distortion: np.ndarray,
) -> BoardObservation | None:
    lookup = {
        int(marker_id): np.asarray(points).reshape(4, 3)
        for points, marker_id in zip(
            board.getObjPoints(), board.getIds().reshape(-1), strict=True
        )
    }
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is not None:
        kept = [
            (corner, int(marker_id))
            for corner, marker_id in zip(corners, ids.reshape(-1), strict=True)
            if int(marker_id) in lookup
        ]
        corners = [corner for corner, _ in kept]
        ids = np.asarray([marker_id for _, marker_id in kept], dtype=np.int32)
        ids = ids.reshape(-1, 1) if kept else None
    if ids is not None:
        corners, ids, _, _ = detector.refineDetectedMarkers(
            gray, board, corners, ids, rejected, matrix, distortion
        )
    if ids is None or len(corners) < MIN_TAGS:
        return None
    object_points = np.concatenate(
        [lookup[int(marker_id)] for marker_id in ids.reshape(-1)]
    ).astype(np.float32)
    image_points = np.concatenate([corner.reshape(4, 2) for corner in corners]).astype(
        np.float32
    )
    success, rotation, translation = cv2.solvePnP(
        object_points, image_points, matrix, distortion, flags=cv2.SOLVEPNP_IPPE
    )
    if not success or float(translation.reshape(3)[2]) <= 0:
        return None
    rotation, translation = cv2.solvePnPRefineLM(
        object_points, image_points, matrix, distortion, rotation, translation
    )
    projected = cv2.projectPoints(
        object_points, rotation, translation, matrix, distortion
    )[0].reshape(-1, 2)
    error = float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))
    if error > MAX_REPROJECTION_PX:
        return None
    return BoardObservation(
        transform_from_pose(rotation, translation), corners, ids, error
    )


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    translation = np.mean([transform[:3, 3] for transform in transforms], axis=0)
    rotation_sum = np.sum([transform[:3, :3] for transform in transforms], axis=0)
    left, _, right = np.linalg.svd(rotation_sum)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    average = np.eye(4, dtype=np.float64)
    average[:3, :3] = rotation
    average[:3, 3] = translation
    return average


def transform_errors(
    transforms: list[np.ndarray], estimate: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    translations = np.asarray(
        [np.linalg.norm(transform[:3, 3] - estimate[:3, 3]) for transform in transforms]
    )
    rotations = np.asarray(
        [
            rotation_error_deg(estimate[:3, :3], transform[:3, :3])
            for transform in transforms
        ]
    )
    return translations, rotations


def robust_limit(values: np.ndarray, floor: float) -> float:
    median = float(np.median(values))
    deviation = float(np.median(np.abs(values - median)))
    return max(floor, median + 3.0 * 1.4826 * deviation)


def solve_extrinsics(samples: list[PosePair]) -> CalibrationResult:
    transforms = [
        pair.top.transform @ np.linalg.inv(pair.left.transform) for pair in samples
    ]
    initial = average_transforms(transforms)
    translations, rotations = transform_errors(transforms, initial)
    inliers = (translations <= robust_limit(translations, 0.012)) & (
        rotations <= robust_limit(rotations, 1.5)
    )
    if int(inliers.sum()) < MIN_POSES:
        raise RuntimeError(
            f"Only {int(inliers.sum())}/{len(samples)} calibration poses agree"
        )
    estimate = average_transforms(
        [transform for transform, keep in zip(transforms, inliers, strict=True) if keep]
    )
    translations, rotations = transform_errors(transforms, estimate)
    return CalibrationResult(estimate, translations, rotations, inliers)


def poses_differ(
    first: np.ndarray, second: np.ndarray, translation_m: float, rotation_deg: float
) -> bool:
    translation = np.linalg.norm(first[:3, 3] - second[:3, 3])
    rotation = rotation_error_deg(first[:3, :3], second[:3, :3])
    return bool(translation > translation_m or rotation > rotation_deg)


def update_window(
    window: list[tuple[float, BoardObservation]],
    observation: BoardObservation | None,
    timestamp: float,
) -> None:
    window[:] = [item for item in window if item[0] >= timestamp - 3.0 * STABILITY_S]
    if observation is None:
        return
    if window and poses_differ(
        window[0][1].transform,
        observation.transform,
        MAX_STABILITY_TRANSLATION_M,
        MAX_STABILITY_ROTATION_DEG,
    ):
        window.clear()
    window.append((timestamp, observation))


def average_observations(observations: list[BoardObservation]) -> BoardObservation:
    best = max(
        observations, key=lambda item: (len(item.corners), -item.reprojection_error)
    )
    return BoardObservation(
        average_transforms([item.transform for item in observations]),
        best.corners,
        best.ids,
        float(np.median([item.reprojection_error for item in observations])),
    )


def stable_candidate(
    top_window: list[tuple[float, BoardObservation]],
    left_window: list[tuple[float, BoardObservation]],
) -> tuple[PosePair | None, float]:
    if len(top_window) < 3 or len(left_window) < 3:
        return None, 0.0
    start = max(top_window[0][0], left_window[0][0])
    end = min(top_window[-1][0], left_window[-1][0])
    progress = float(np.clip((end - start) / STABILITY_S, 0.0, 1.0))
    if progress < 1.0:
        return None, progress
    top = [item for seen, item in top_window if seen >= start]
    left = [item for seen, item in left_window if seen >= start]
    if len(top) < 3 or len(left) < 3:
        return None, progress
    return PosePair(average_observations(top), average_observations(left)), progress


def is_new_pose(candidate: PosePair, samples: list[PosePair]) -> bool:
    return all(
        poses_differ(
            candidate.top.transform,
            sample.top.transform,
            MIN_TRANSLATION_M,
            MIN_ROTATION_DEG,
        )
        for sample in samples
    )


def draw_view(
    image: np.ndarray, observation: BoardObservation | None, name: str
) -> np.ndarray:
    if observation is None:
        color, detail = (70, 90, 240), "board not found"
    else:
        color = (70, 220, 110)
        detail = (
            f"{len(observation.corners)} tags | {observation.reprojection_error:.2f} px"
        )
        cv2.aruco.drawDetectedMarkers(
            image, observation.corners, observation.ids.reshape(-1, 1), color
        )
    cv2.putText(
        image, f"{name} | {detail}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
    )
    return image


def stream_color(rs, serial: str, width: int, height: int, fps: int, warmup: int):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)
    try:
        intrinsics = (
            profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
        )
        for _ in range(warmup):
            pipeline.wait_for_frames(3000)
    except Exception:
        pipeline.stop()
        raise
    return pipeline, intrinsics


def start_camera(
    rs, serial: str, width: int = WIDTH, height: int = HEIGHT, fps: int = FPS
):
    try:
        return stream_color(rs, serial, width, height, fps, 20)
    except RuntimeError as error:
        if "didn't arrive" not in str(error):
            raise
    print(f"Camera {serial} stalled; kicking it with a low-resolution stream")
    stream_color(rs, serial, 640, 480, 30, 5)[0].stop()
    return stream_color(rs, serial, width, height, fps, 20)


def next_color(pipeline) -> np.ndarray:
    frame = pipeline.wait_for_frames(3000).get_color_frame()
    if not frame:
        raise RuntimeError("Camera did not return a color frame")
    return np.asanyarray(frame.get_data()).copy()


def capture_samples(board) -> tuple[list[PosePair], Any, Any]:
    rs: Any = importlib.import_module("pyrealsense2")
    detector = detector_from_dictionary(board.getDictionary())
    top_pipeline, top_intrinsics = start_camera(rs, TOP_CAMERA_SERIAL)
    try:
        left_pipeline, left_intrinsics = start_camera(rs, LEFT_CAMERA_SERIAL)
    except Exception:
        top_pipeline.stop()
        raise
    top_matrix, top_distortion = intrinsics_values(top_intrinsics)
    left_matrix, left_distortion = intrinsics_values(left_intrinsics)
    samples: list[PosePair] = []
    top_window: list[tuple[float, BoardObservation]] = []
    left_window: list[tuple[float, BoardObservation]] = []
    window = "Camera calibration | top and left wrist"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1800, 506)
    print("Show the board to both cameras, tilted; hold still for each pose")
    print("Press Enter to finish early, Esc to cancel")
    try:
        while len(samples) < POSES:
            top_image = next_color(top_pipeline)
            left_image = next_color(left_pipeline)
            top = observe_board(top_image, board, detector, top_matrix, top_distortion)
            left = observe_board(
                left_image, board, detector, left_matrix, left_distortion
            )
            now = time.monotonic()
            update_window(top_window, top, now)
            update_window(left_window, left, now)
            candidate, progress = stable_candidate(top_window, left_window)
            if candidate is not None:
                top_window.clear()
                left_window.clear()
                if is_new_pose(candidate, samples):
                    samples.append(candidate)
                    print(f"Captured pose {len(samples)}; move and tilt the board")
            combined = np.hstack(
                (draw_view(top_image, top, "TOP"), draw_view(left_image, left, "LEFT"))
            )
            cv2.putText(
                combined,
                f"poses {len(samples)}/{POSES} | hold still {round(100 * progress)}%",
                (14, combined.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (84, 220, 139),
                2,
            )
            cv2.imshow(window, combined)
            key = cv2.waitKey(10) & 0xFF
            if key in (10, 13) and len(samples) >= MIN_POSES:
                break
            if key == 27:
                raise RuntimeError("Calibration cancelled")
    finally:
        cv2.destroyWindow(window)
        top_pipeline.stop()
        left_pipeline.stop()
    return samples, top_intrinsics, left_intrinsics


def intrinsics_dict(intrinsics) -> dict[str, object]:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "model": str(intrinsics.model).split(".")[-1],
        "coeffs": [float(value) for value in intrinsics.coeffs],
    }


def save_calibration(
    result: CalibrationResult, samples: list[PosePair], top_intrinsics, left_intrinsics
) -> None:
    translations = result.translation_errors_m[result.inliers] * 1000.0
    rotations = result.rotation_errors_deg[result.inliers]
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_frame": "top_camera_optical_frame",
        "source_frame": "left_wrist_camera_optical_frame",
        "top_serial": TOP_CAMERA_SERIAL,
        "left_serial": LEFT_CAMERA_SERIAL,
        "resolution": [WIDTH, HEIGHT],
        "board": {
            "dictionary": DICTIONARY,
            "marker_mm": MARKER_MM,
            "tags": {
                str(marker_id): list(tag) for marker_id, tag in BOARD_TAGS.items()
            },
        },
        "top_from_left": result.transform.tolist(),
        "samples": len(samples),
        "inliers": int(result.inliers.sum()),
        "translation_residual_mm": {
            "median": float(np.median(translations)),
            "maximum": float(np.max(translations)),
        },
        "rotation_residual_deg": {
            "median": float(np.median(rotations)),
            "maximum": float(np.max(rotations)),
        },
        "top_intrinsics": intrinsics_dict(top_intrinsics),
        "left_intrinsics": intrinsics_dict(left_intrinsics),
        "per_pose": [
            {
                "inlier": bool(result.inliers[index]),
                "translation_residual_mm": float(
                    result.translation_errors_m[index] * 1000.0
                ),
                "rotation_residual_deg": float(result.rotation_errors_deg[index]),
                "top_reprojection_px": sample.top.reprojection_error,
                "left_reprojection_px": sample.left.reprojection_error,
            }
            for index, sample in enumerate(samples)
        ],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_name(f".{OUTPUT_PATH.name}.{os.getpid()}.tmp")
    temporary.write_bytes(json.dumps(payload, indent=2).encode())
    os.replace(temporary, OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")
    print(
        f"translation residual: median {np.median(translations):.1f} mm, "
        f"max {np.max(translations):.1f} mm"
    )
    print(
        f"rotation residual: median {np.median(rotations):.2f} deg, "
        f"max {np.max(rotations):.2f} deg"
    )


def main() -> None:
    board = board_from_placements(
        BOARD_TAGS, MARKER_MM, dictionary_from_name(DICTIONARY)
    )
    samples, top_intrinsics, left_intrinsics = capture_samples(board)
    result = solve_extrinsics(samples)
    translations = result.translation_errors_m[result.inliers]
    rotations = result.rotation_errors_deg[result.inliers]
    if float(np.max(translations)) > MAX_TRANSLATION_RESIDUAL_M:
        raise RuntimeError("Translation residual is too high; repeat calibration")
    if float(np.max(rotations)) > MAX_ROTATION_RESIDUAL_DEG:
        raise RuntimeError("Rotation residual is too high; repeat calibration")
    save_calibration(result, samples, top_intrinsics, left_intrinsics)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
