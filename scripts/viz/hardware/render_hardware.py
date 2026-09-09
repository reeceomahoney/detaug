import shlex
import subprocess
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import draccus
from PIL import Image

GROUPS = [
    ("IMG_4556", 200, (28, 74, 364)),
    ("IMG_4557", 220, (342, 42, 305)),
    ("IMG_4559", 200, (120, 191, 234)),
    ("IMG_4560", 350, (189, 333, 98)),
]


@dataclass
class Config:
    videos: Path = Path.home() / "Downloads/drive-videos"
    out: Path = Path("outputs/hardware_demos.pdf")
    dpi: int = 600
    tile: tuple[int, int] = field(default_factory=lambda: (640, 480))
    crop: tuple[int, int] = field(default_factory=lambda: (1440, 1080))
    inner: int = 6
    outer: int = 26


def frame(path: Path, t: float, x: int, crop: tuple[int, int]):
    cw, ch = crop
    args = f"-v error -ss {t} -i {shlex.quote(str(path))} -frames:v 1"
    args += f" -vf crop={cw}:{ch}:{x}:0 -f image2pipe -c:v png -"
    png = subprocess.run(
        ["ffmpeg", *shlex.split(args)], capture_output=True, check=True
    )
    return Image.open(BytesIO(png.stdout))


def save(fig: Image.Image, out: Path, dpi: int):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".pdf":
        fig.save(out, resolution=dpi)
    else:
        fig.save(out, dpi=(dpi, dpi))
    print(f"{out} {fig.size}")


@draccus.wrap()
def main(cfg: Config):
    tw, th = cfg.tile
    groups = [
        [
            frame(cfg.videos / f"{name}.MOV", t, x, cfg.crop).resize(
                cfg.tile, Image.Resampling.LANCZOS
            )
            for t in ts
        ]
        for name, x, ts in GROUPS
    ]

    gw = 3 * tw + 2 * cfg.inner
    fig = Image.new("RGB", (2 * gw + cfg.outer, 2 * th + cfg.outer), "white")
    for gi, tiles in enumerate(groups):
        gx, gy = (gi % 2) * (gw + cfg.outer), (gi // 2) * (th + cfg.outer)
        for ti, tile in enumerate(tiles):
            fig.paste(tile, (gx + ti * (tw + cfg.inner), gy))

    save(fig, cfg.out, cfg.dpi)


if __name__ == "__main__":
    main()
