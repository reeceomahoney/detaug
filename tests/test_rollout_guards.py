import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rollout import check_box, freeze_box  # noqa: E402  # ty: ignore[unresolved-import]

OBJECT = np.array([0.350, -0.023, 0.095, 0.059, 0.058, 0.123], np.float32)
PHANTOM = np.array([0.585, 0.152, 0.082, 0.061, 0.060, 0.116], np.float32)
TABLE = np.array([0.453, -0.060, 0.011, 0.308, 0.308, 0.047], np.float32)


class Replay:
    def __init__(self, boxes):
        self.boxes = boxes
        self.index = 0

    def latest(self):
        box = self.boxes[min(self.index, len(self.boxes) - 1)]
        self.index += 1
        return None if box is None else {"box": box.tolist(), "stamp": time.time()}


def test_a_still_obstacle_freezes_to_itself():
    box, spread, count = freeze_box(Replay([OBJECT] * 40), 0.3)
    assert np.allclose(box, OBJECT, atol=1e-6)
    assert spread < 1e-6 and count > 1
    check_box(box, 0.15)


def test_a_jumping_tracker_shows_up_as_spread():
    _, spread, _ = freeze_box(Replay(([OBJECT] * 5 + [PHANTOM] * 5) * 4), 0.3)
    assert spread > 0.02


def test_a_table_sized_box_is_rejected():
    with pytest.raises(RuntimeError, match="footprint"):
        check_box(TABLE, 0.15)


def test_a_degenerate_box_is_rejected():
    with pytest.raises(RuntimeError, match="degenerate"):
        check_box(np.array([0.3, 0.0, 0.1, 0.05, 0.05, 0.0], np.float32), 0.15)


def test_an_empty_window_is_rejected():
    with pytest.raises(RuntimeError, match="no obstacle"):
        freeze_box(Replay([None]), 0.2)
