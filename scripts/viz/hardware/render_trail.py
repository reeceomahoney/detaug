import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import draccus
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent))
from render_hardware import frame, save  # noqa: E402


@dataclass
class Config:
    videos: Path = Path.home() / "Downloads/drive-videos"
    video: str = "IMG_4556"
    out: Path = Path("outputs/hardware_trail.pdf")
    masks: Path = Path("outputs/trail_masks")
    track: Path = Path("outputs/trail_track.json")
    base: float = 403.0
    times: list[float] = field(default_factory=lambda: [405.3, 406.4, 407.3, 408.0])
    crop: tuple[int, int] = field(default_factory=lambda: (1080, 1080))
    crop_x: int = 700
    alpha: tuple[float, float] = field(default_factory=lambda: (0.4, 1.0))
    trim_bottom: int = 20
    feather: float = 2.0
    dot: int = 9
    fill_mask: Path = Path("outputs/trail_fill.png")
    fill: str = ""
    fill_src: str = "390,800,780,1080"
    fill_feather: int = 30
    decoy_via: str = "480,620;520,545;480,780"
    decoy_trim: int = 4
    decoy_rgb: tuple[int, int, int] = field(default_factory=lambda: (150, 150, 150))
    dot_rgb: tuple[int, int, int] = field(default_factory=lambda: (40, 110, 255))
    dpi: int = 600


def mask_path(cfg: Config, t: float) -> Path:
    return cfg.masks / f"{cfg.video}_{t}.png"


@draccus.wrap()
def main(cfg: Config):
    path = cfg.videos / f"{cfg.video}.MOV"

    def grab(t: float):
        return np.asarray(frame(path, t, cfg.crop_x, cfg.crop))

    plate = grab(cfg.base)
    canvas = plate.astype(float)
    for w, t in zip(np.linspace(*cfg.alpha, len(cfg.times)), cfg.times):
        p = mask_path(cfg, t)
        if not p.exists():
            raise SystemExit(f"no mask at {p}, run scripts/viz/edit_mask.py")
        m = np.asarray(Image.open(p).convert("L")) > 127
        a = (w * cv2.GaussianBlur(m.astype(float), (0, 0), cfg.feather))[..., None]
        canvas = canvas * (1 - a) + grab(t).astype(float) * a
        print(f"t={t} alpha={w:.2f} px={int(m.sum())}")

    def dots(xy, rgb):
        for x, y in xy:
            cv2.circle(
                fig, (int(x), int(y)), cfg.dot + 2, (255, 255, 255), -1, cv2.LINE_AA
            )
            cv2.circle(fig, (int(x), int(y)), cfg.dot, rgb, -1, cv2.LINE_AA)

    fig = canvas.astype(np.uint8)
    region = None
    if cfg.fill_mask.exists():
        region = np.asarray(Image.open(cfg.fill_mask).convert("L")) > 127
    elif cfg.fill:
        x0, y0, x1, y1 = (int(v) for v in cfg.fill.split(","))
        region = np.zeros(fig.shape[:2], bool)
        region[y0:y1, x0:x1] = True

    if region is not None and region.any():
        ys, xs = np.where(region)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sx0, sy0, sx1, sy1 = (int(v) for v in cfg.fill_src.split(","))
        patch = cv2.resize(plate[sy0:sy1, sx0:sx1], (x1 - x0, y1 - y0))
        al = cv2.GaussianBlur(
            region[y0:y1, x0:x1].astype(np.float32), (0, 0), cfg.fill_feather / 4
        )[..., None]
        fig[y0:y1, x0:x1] = (fig[y0:y1, x0:x1] * (1 - al) + patch * al).astype(np.uint8)
        print(f"fill {int(region.sum())} px")
    if cfg.track.exists():
        pts = json.loads(cfg.track.read_text())
        a = np.array(pts[cfg.decoy_trim][1:])
        b = np.array(pts[-1 - cfg.decoy_trim][1:])
        u = np.linspace(0, 1, len(pts) - 2 * cfg.decoy_trim)[:, None]
        for via in filter(None, cfg.decoy_via.split(";")):
            mid = np.array([float(v) for v in via.split(",")])
            c = 2 * mid - (a + b) / 2
            dots((1 - u) ** 2 * a + 2 * (1 - u) * u * c + u**2 * b, cfg.decoy_rgb)
        dots([q[1:] for q in pts], cfg.dot_rgb)
        print(f"{len(pts)} track dots")

    img = Image.fromarray(fig[: fig.shape[0] - cfg.trim_bottom])
    save(img, cfg.out, cfg.dpi)
    if cfg.out.suffix != ".png":
        save(img, cfg.out.with_suffix(".png"), cfg.dpi)


if __name__ == "__main__":
    main()
