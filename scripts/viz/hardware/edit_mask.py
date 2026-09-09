import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import draccus
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

sys.path.append(str(Path(__file__).parent))
from render_hardware import frame  # noqa: E402
from render_trail import Config as TrailConfig  # noqa: E402
from render_trail import mask_path  # noqa: E402

HELP = "left add | right remove | u undo | c clear | r reset | n/p pose | q quit"


@dataclass
class Config(TrailConfig):
    model: str = "facebook/sam-vit-huge"
    scale: float = 0.55
    click_floor: int = 200
    tint: tuple[int, int, int] = field(default_factory=lambda: (0, 0, 255))


def point_mask(img, emb, proc, model, xy, floor: int):
    inp = proc(Image.fromarray(img), input_points=[[list(xy)]], return_tensors="pt").to(
        model.device
    )
    inp.pop("pixel_values")
    with torch.no_grad():
        out = model(**inp, image_embeddings=emb, multimask_output=True)
    cand = proc.image_processor.post_process_masks(
        out.pred_masks.cpu(),
        inp["original_sizes"].cpu(),
        inp["reshaped_input_sizes"].cpu(),
    )[0][0].numpy()
    areas = [int(c.sum()) for c in cand]
    ok = [i for i, a in enumerate(areas) if a >= floor]
    return cand[min(ok or range(len(cand)), key=lambda i: areas[i])]


def overlay(img, mask, tint, scale):
    vis = img.copy()
    vis[mask] = (0.45 * np.array(tint) + 0.55 * vis[mask]).astype(np.uint8)
    edge = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    )
    vis[edge > 0] = tint
    return cv2.resize(vis[:, :, ::-1], None, fx=scale, fy=scale)


@draccus.wrap()
def main(cfg: Config):
    path = cfg.videos / f"{cfg.video}.MOV"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = SamProcessor.from_pretrained(cfg.model)
    model = SamModel.from_pretrained(cfg.model).to(device=dev).eval()
    cfg.masks.mkdir(parents=True, exist_ok=True)

    click = {}
    cv2.namedWindow("mask", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(
        "mask",
        lambda e, x, y, *_: (
            click.update(xy=(x, y), add=e == cv2.EVENT_LBUTTONDOWN)
            if e in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN)
            else None
        ),
    )

    i = 0
    while 0 <= i < len(cfg.times):
        t = cfg.times[i]
        img = np.asarray(frame(path, t, cfg.crop_x, cfg.crop))
        out = mask_path(cfg, t)
        blank = np.zeros(img.shape[:2], bool)
        mask = np.asarray(Image.open(out).convert("L")) > 127 if out.exists() else blank
        start, undo = mask.copy(), []

        with torch.no_grad():
            emb = model.get_image_embeddings(
                proc(Image.fromarray(img), return_tensors="pt").to(dev)["pixel_values"]
            )

        print(f"[{i + 1}/{len(cfg.times)}] t={t}  {HELP}")
        step = 0
        while True:
            cv2.imshow("mask", overlay(img, mask, cfg.tint, cfg.scale))
            key = cv2.waitKey(20) & 0xFF
            if click:
                xy = [int(v / cfg.scale) for v in click["xy"]]
                seg = point_mask(img, emb, proc, model, xy, cfg.click_floor)
                undo.append(mask.copy())
                mask = mask | seg if click.pop("add") else mask & ~seg
                click.clear()
                print(f"  {'+' if undo else ''}{int(mask.sum())} px")
            if key == ord("u") and undo:
                mask = undo.pop()
            elif key == ord("c"):
                undo.append(mask.copy())
                mask = blank.copy()
            elif key == ord("r"):
                mask, undo = start.copy(), []
            elif key in (ord("n"), ord("p"), ord("q")):
                Image.fromarray((mask * 255).astype(np.uint8)).save(out)
                print(f"  saved {out}")
                step = {ord("n"): 1, ord("p"): -1, ord("q"): len(cfg.times)}[key]
                break
        i += step
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
