import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import draccus
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent))
from render_hardware import frame  # noqa: E402
from render_trail import Config as TrailConfig  # noqa: E402

HELP = "click corners | u undo | c clear | s save | q quit"


@dataclass
class Config(TrailConfig):
    fill_mask: Path = Path("outputs/trail_fill.png")
    scale: float = 0.55
    tint: tuple[int, int, int] = field(default_factory=lambda: (0, 0, 255))


@draccus.wrap()
def main(cfg: Config):
    path = cfg.videos / f"{cfg.video}.MOV"
    img = np.asarray(frame(path, cfg.base, cfg.crop_x, cfg.crop))[:, :, ::-1].copy()
    pts: list[list[int]] = []

    click = {}
    cv2.namedWindow("fill", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(
        "fill",
        lambda e, x, y, *_: (
            click.update(xy=(x, y)) if e == cv2.EVENT_LBUTTONDOWN else None
        ),
    )

    print(f"{HELP}")
    while True:
        vis = img.copy()
        if pts:
            poly = np.array(pts, np.int32)
            if len(pts) > 2:
                shade = vis.copy()
                cv2.fillPoly(shade, [poly], cfg.tint)
                vis = cv2.addWeighted(vis, 0.6, shade, 0.4, 0)
            cv2.polylines(vis, [poly], len(pts) > 2, cfg.tint, 3)
            for x, y in pts:
                cv2.circle(vis, (x, y), 7, (255, 255, 255), -1)
                cv2.circle(vis, (x, y), 5, cfg.tint, -1)
        cv2.imshow("fill", cv2.resize(vis, None, fx=cfg.scale, fy=cfg.scale))

        key = cv2.waitKey(20) & 0xFF
        if click:
            pts.append([int(v / cfg.scale) for v in click.pop("xy")])
            click.clear()
            print(f"  {len(pts)} corners")
        if key == ord("u") and pts:
            pts.pop()
        elif key == ord("c"):
            pts.clear()
        elif key == ord("s"):
            if len(pts) < 3:
                print("  need at least 3 corners")
                continue
            mask = np.zeros(img.shape[:2], np.uint8)
            cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
            cfg.fill_mask.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(cfg.fill_mask)
            print(f"saved {cfg.fill_mask} ({int((mask > 0).sum())} px)")
            break
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
