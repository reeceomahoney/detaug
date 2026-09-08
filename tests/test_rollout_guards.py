import logging
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rollout import (  # noqa: E402  # ty: ignore[unresolved-import]
    DetAugConfig,
    ObstacleGuard,
    diagnose,
    recalibrate,
    settle_box,
)

OBJECT = np.array([0.350, -0.023, 0.095, 0.059, 0.058, 0.123], np.float32)
PHANTOM = np.array([0.585, 0.152, 0.082, 0.061, 0.060, 0.116], np.float32)
TABLE = np.array([0.453, -0.060, 0.011, 0.308, 0.308, 0.047], np.float32)
FLAT = np.array([0.3, 0.0, 0.1, 0.05, 0.05, 0.0], np.float32)


def config(**overrides) -> DetAugConfig:
    values = {"settle": 0.15, "recal_timeout": 0.5, "recal_attempts": 2}
    values.update(overrides)
    return DetAugConfig(**values)


class FakeTracker:
    def __init__(self, boxes, tracking=("top", "left"), realign=None):
        self.boxes = list(boxes)
        self.index = 0
        self.tracking = set(tracking)
        self.realign = realign
        self.requests: list[float] = []
        self.aligned_at = 0.0
        self.phase = "Tracking both camera views"

    def latest(self):
        box = self.boxes[min(self.index, len(self.boxes) - 1)]
        self.index += 1
        if box is None:
            return None
        return {"box": box.tolist(), "stamp": time.time()}

    def box(self):
        payload = self.latest()
        return None if payload is None else np.asarray(payload["box"], np.float32)

    def status(self):
        return {
            "phase": self.phase,
            "cameras": {
                name: {"tracking": name in self.tracking, "aligned_at": self.aligned_at}
                for name in ("top", "left")
            },
        }

    def recalibrate(self):
        stamp = time.time()
        self.requests.append(stamp)
        if self.realign is not None:
            self.realign(self)
        return stamp


def test_a_plausible_box_passes():
    assert diagnose(OBJECT, OBJECT, 0.15, 0.05) is None


def test_a_table_sized_box_is_wrong():
    assert "footprint" in str(diagnose(TABLE, None, 0.15, 0.05))


def test_a_degenerate_box_is_wrong():
    assert "degenerate" in str(diagnose(FLAT, None, 0.15, 0.05))


def test_a_missing_box_is_wrong():
    assert "no obstacle" in str(diagnose(None, OBJECT, 0.15, 0.05))


def test_a_teleporting_centre_is_wrong():
    assert "jumped" in str(diagnose(PHANTOM, OBJECT, 0.15, 0.05))


def test_a_still_tracked_obstacle_settles():
    box, reason = settle_box(FakeTracker([OBJECT] * 40), config())
    assert reason is None
    assert np.allclose(box, OBJECT)


def test_a_flickering_tracker_does_not_settle():
    _, reason = settle_box(FakeTracker(([OBJECT] * 3 + [PHANTOM] * 3) * 20), config())
    assert "wandered" in str(reason)


def test_a_leftover_box_with_no_tracking_view_does_not_settle():
    _, reason = settle_box(FakeTracker([OBJECT] * 40, tracking=()), config())
    assert "leftover" in str(reason)


def test_an_empty_feed_does_not_settle():
    box, reason = settle_box(FakeTracker([None]), config())
    assert box is None and "no obstacle" in str(reason)


def test_recalibrate_waits_for_both_views_to_realign():
    def realign(tracker):
        tracker.aligned_at = time.time()

    assert recalibrate(FakeTracker([OBJECT], realign=realign), 0.5) is None


def test_recalibrate_reports_a_failed_realignment():
    def realign(tracker):
        tracker.phase = "Realignment failed: match 0.41 is below 0.6"

    failure = recalibrate(FakeTracker([OBJECT], realign=realign), 0.5)
    assert "Realignment failed" in str(failure)


def test_recalibrate_times_out_when_nothing_happens():
    assert "did not realign" in str(recalibrate(FakeTracker([OBJECT]), 0.2))


def test_startup_recalibrates_a_wrong_box_and_accepts_the_fix():
    def realign(tracker):
        tracker.boxes = [OBJECT]
        tracker.aligned_at = time.time()

    tracker = FakeTracker([TABLE], realign=realign)
    guard = ObstacleGuard(tracker, config())
    box = guard.startup()
    assert np.allclose(box, OBJECT)
    assert len(tracker.requests) == 1 and guard.recalibrations == 1


def test_startup_gives_up_after_the_attempt_budget():
    def realign(tracker):
        tracker.aligned_at = time.time()

    tracker = FakeTracker([TABLE], realign=realign)
    with pytest.raises(RuntimeError, match="after 2 recalibrations"):
        ObstacleGuard(tracker, config(recal_attempts=2)).startup()
    assert len(tracker.requests) == 2


def test_a_jump_during_the_rollout_holds_the_box_until_the_tracker_realigns(caplog):
    def realign(tracker):
        tracker.boxes = [OBJECT]
        tracker.aligned_at = time.time()

    tracker = FakeTracker([OBJECT] * 3 + [PHANTOM] * 3, realign=realign)
    guard = ObstacleGuard(tracker, config(settle=0.0))
    guard.reference = OBJECT
    assert np.allclose(guard.current(), OBJECT)
    guard.reference = OBJECT
    tracker.index = 3
    with caplog.at_level(logging.WARNING):
        assert guard.current() is None
    assert "jumped" in caplog.text
    assert len(tracker.requests) == 1
    assert np.allclose(guard.current(), OBJECT)
    assert guard.requested == 0.0


def test_a_failed_realignment_keeps_the_last_box_and_backs_off():
    def realign(tracker):
        tracker.phase = "Realignment failed: match 0.41 is below 0.6"

    tracker = FakeTracker([PHANTOM], realign=realign)
    guard = ObstacleGuard(tracker, config(settle=0.0, recal_timeout=10.0))
    guard.reference = OBJECT
    assert guard.current() is None
    assert guard.current() is None
    assert guard.current() is None
    assert len(tracker.requests) == 1
