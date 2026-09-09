import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import draccus
import numpy as np
from scipy.interpolate import CubicSpline

sys.path.append(str(Path(__file__).parent))
from render_hardware import frame  # noqa: E402
from render_trail import Config as TrailConfig  # noqa: E402

HELP = "click the gripper | b back | s skip | q done"


@dataclass
class Config(TrailConfig):
    track: Path = Path("outputs/trail_track.json")
    start: float = 405.2
    end: float = 408.0
    key_step: float = 0.5
    step: float = 0.1
    scale: float = 0.55


@draccus.wrap()
def main(cfg: Config):
    path = cfg.videos / f"{cfg.video}.MOV"
    keys = list(np.arange(cfg.start, cfg.end + 1e-6, cfg.key_step))
    picked: dict[float, tuple[float, float]] = {}

    click = {}
    cv2.namedWindow("track", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(
        "track",
        lambda e, x, y, *_: (
            click.update(xy=(x, y)) if e == cv2.EVENT_LBUTTONDOWN else None
        ),
    )

    i = 0
    while 0 <= i < len(keys):
        t = float(keys[i])
        img = np.asarray(frame(path, t, cfg.crop_x, cfg.crop))[:, :, ::-1].copy()
        for u, (x, y) in picked.items():
            cv2.circle(img, (int(x), int(y)), 6, (60, 60, 255), -1 if u != t else 2)
        print(f"[{i + 1}/{len(keys)}] t={t:.2f}  {HELP}")
        while True:
            cv2.imshow("track", cv2.resize(img, None, fx=cfg.scale, fy=cfg.scale))
            key = cv2.waitKey(20) & 0xFF
            if click:
                picked[t] = tuple(v / cfg.scale for v in click.pop("xy"))
                click.clear()
                i += 1
                break
            if key == ord("b"):
                i -= 1
                break
            if key == ord("s"):
                i += 1
                break
            if key == ord("q"):
                i = len(keys)
                break
    cv2.destroyAllWindows()

    if len(picked) < 3:
        raise SystemExit(f"need at least 3 points, got {len(picked)}")
    ts = sorted(picked)
    fit = CubicSpline(ts, [picked[t] for t in ts])
    dense = np.arange(ts[0], ts[-1] + 1e-6, cfg.step)
    pts = [[round(float(t), 3), *map(float, fit(t))] for t in dense]

    cfg.track.parent.mkdir(parents=True, exist_ok=True)
    cfg.track.write_text(json.dumps(pts))
    print(f"{cfg.track} {len(picked)} clicks -> {len(pts)} points")


if __name__ == "__main__":
    main()
