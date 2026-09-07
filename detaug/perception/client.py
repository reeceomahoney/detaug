from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np


class ObstacleClient:
    def __init__(
        self,
        url: str,
        period: float = 0.2,
        timeout: float = 1.0,
        stale: float = 3.0,
    ):
        self.url = url
        self.period = period
        self.timeout = timeout
        self.stale = stale
        self.lock = threading.Lock()
        self.payload: dict | None = None
        self.received = 0.0
        self.error: str | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def fetch(self) -> dict:
        with urllib.request.urlopen(self.url, timeout=self.timeout) as response:
            return json.loads(response.read())

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                payload = self.fetch()
                with self.lock:
                    self.payload = payload
                    self.received = time.monotonic()
                    self.error = None
            except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
                with self.lock:
                    self.error = str(error)
            self.stop_event.wait(self.period)

    def latest(self) -> dict | None:
        with self.lock:
            if self.payload is None:
                return None
            if time.monotonic() - self.received > self.stale:
                return None
            return self.payload

    def box(self) -> np.ndarray | None:
        payload = self.latest()
        if payload is None or payload.get("box") is None:
            return None
        return np.asarray(payload["box"], np.float32)

    def cloud(self) -> np.ndarray | None:
        payload = self.latest()
        if payload is None or not payload.get("cloud"):
            return None
        return np.asarray(payload["cloud"], np.float32)

    def wait(self, timeout: float) -> dict | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            payload = self.latest()
            if payload is not None and payload.get("box") is not None:
                return payload
            time.sleep(0.1)
        return None
