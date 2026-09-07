from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FILES = {
    "top_raw.jpg": "image/jpeg",
    "top_overlay.jpg": "image/jpeg",
    "top_cloud.jpg": "image/jpeg",
    "top_mask.png": "image/png",
    "left_raw.jpg": "image/jpeg",
    "left_overlay.jpg": "image/jpeg",
    "left_cloud.jpg": "image/jpeg",
    "scene_cloud.jpg": "image/jpeg",
    "left_mask.png": "image/png",
}
REFERENCE_FILES = {"top_reference.png", "left_reference.png"}
DASHBOARD_DIR = Path(__file__).parent / "dashboard"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/dashboard.css": ("dashboard.css", "text/css; charset=utf-8"),
    "/dashboard.js": ("dashboard.js", "text/javascript; charset=utf-8"),
}


class LiveState:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.images: dict[str, tuple[int, bytes]] = {}
        self.versions: dict[str, int] = {}
        self.samples: dict[str, tuple[int, dict[str, object]]] = {}
        self.sample_versions: dict[str, int] = {}
        self.gripper_markers: dict[str, tuple[float, float, float]] = {}
        self.camera_boxes: dict[str, tuple[tuple[float, float], ...]] = {}
        self.robot_state_data: tuple[float, ...] | None = None
        self.status = b'{"running":false,"updated_at":0,"cameras":{}}'
        self.realign_requested = False
        self.trim_percent = 40.0
        self.obstacle_data: dict[str, object] | None = None

    def publish_image(self, name: str, payload: bytes) -> None:
        with self.condition:
            version = self.versions.get(name, 0) + 1
            self.versions[name] = version
            self.images[name] = (version, payload)
            self.condition.notify_all()

    def image(self, name: str) -> tuple[int, bytes] | None:
        with self.condition:
            return self.images.get(name)

    def wait_image(
        self, name: str, version: int, timeout: float
    ) -> tuple[int, bytes] | None:
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.versions.get(name, 0) <= version:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
            return self.images.get(name)

    def publish_status(self, status: Mapping[str, object]) -> None:
        encoded = json.dumps(status).encode()
        with self.condition:
            self.status = encoded

    def publish_sample(self, name: str, sample: dict[str, object]) -> None:
        with self.condition:
            version = self.sample_versions.get(name, 0) + 1
            self.sample_versions[name] = version
            self.samples[name] = (version, sample)
            self.condition.notify_all()

    def sample(self, name: str) -> tuple[int, dict[str, object]] | None:
        with self.condition:
            return self.samples.get(name)

    def publish_gripper_marker(
        self,
        name: str,
        marker: tuple[float, float, float] | None,
    ) -> None:
        with self.condition:
            if marker is None:
                self.gripper_markers.pop(name, None)
            else:
                self.gripper_markers[name] = marker

    def gripper_marker(self, name: str) -> tuple[float, float, float] | None:
        with self.condition:
            return self.gripper_markers.get(name)

    def publish_camera_box(
        self,
        name: str,
        box: tuple[tuple[float, float], ...] | None,
    ) -> None:
        with self.condition:
            if box is None:
                self.camera_boxes.pop(name, None)
            else:
                self.camera_boxes[name] = box

    def camera_box(
        self,
        name: str,
    ) -> tuple[tuple[float, float], ...] | None:
        with self.condition:
            return self.camera_boxes.get(name)

    def publish_robot_state(self, state: tuple[float, ...]) -> None:
        with self.condition:
            self.robot_state_data = state
            self.condition.notify_all()

    def wait_robot_state(
        self,
        timeout: float,
    ) -> tuple[float, ...] | None:
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.robot_state_data is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
            return self.robot_state_data

    def status_bytes(self) -> bytes:
        with self.condition:
            return self.status

    def publish_obstacle(self, payload: Mapping[str, object] | None) -> None:
        with self.condition:
            self.obstacle_data = None if payload is None else dict(payload)
            self.condition.notify_all()

    def obstacle_bytes(self) -> bytes:
        with self.condition:
            payload = self.obstacle_data
        return json.dumps(payload or {"box": None, "stamp": 0.0}).encode()

    def request_realign(self) -> None:
        with self.condition:
            self.realign_requested = True

    def consume_realign_request(self) -> bool:
        with self.condition:
            requested = self.realign_requested
            self.realign_requested = False
            return requested

    def set_outlier_trim_percent(self, percentage: float) -> None:
        with self.condition:
            self.trim_percent = percentage

    def outlier_trim_percent(self) -> float:
        with self.condition:
            return self.trim_percent


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    live_state: LiveState
    target_dir: Path

    def handle_error(self, request, client_address) -> None:
        del request, client_address


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: DashboardServer

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        static = STATIC_FILES.get(path)
        if static is not None:
            name, content_type = static
            self.send_bytes((DASHBOARD_DIR / name).read_bytes(), content_type)
            return
        if path == "/status.json":
            self.send_bytes(self.server.live_state.status_bytes(), "application/json")
            return
        if path == "/obstacle.json":
            self.send_bytes(self.server.live_state.obstacle_bytes(), "application/json")
            return
        if path.startswith("/stream/"):
            name = path.removeprefix("/stream/")
            if FILES.get(name) != "image/jpeg":
                self.send_error(HTTPStatus.NOT_FOUND)
            else:
                self.stream_jpeg(name)
            return
        if path.startswith("/targets/"):
            name = path.removeprefix("/targets/")
            source = self.server.target_dir / name
            if name not in REFERENCE_FILES or not source.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
            else:
                self.send_bytes(source.read_bytes(), "image/png")
            return
        if path.startswith("/data/"):
            name = path.removeprefix("/data/")
            content_type = FILES.get(name)
            current = self.server.live_state.image(name)
            if content_type is None or current is None:
                self.send_error(HTTPStatus.NOT_FOUND)
            else:
                self.send_bytes(current[1], content_type)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/actions/recalibrate":
            self.server.live_state.request_realign()
            self.send_bytes(b'{"accepted":true}', "application/json")
            return
        if path == "/actions/outlier-trim":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                percentage = float(payload["percentage"])
                if not 0.0 <= percentage <= 50.0:
                    raise ValueError
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid trim percentage")
                return
            self.server.live_state.set_outlier_trim_percent(percentage)
            self.send_bytes(
                json.dumps({"percentage": percentage}).encode(),
                "application/json",
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def stream_jpeg(self, name: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        version = -1
        try:
            while True:
                current = self.server.live_state.wait_image(name, version, 1.0)
                if current is None:
                    continue
                version, payload = current
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode())
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        del format, args


def start_dashboard(
    live_state: LiveState,
    target_dir: Path,
    bind: str,
    port: int,
) -> DashboardServer:
    server = DashboardServer((bind, port), Handler)
    server.live_state = live_state
    server.target_dir = target_dir
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
