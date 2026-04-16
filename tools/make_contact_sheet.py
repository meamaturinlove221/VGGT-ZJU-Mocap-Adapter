import argparse
import os
from typing import List, Tuple

from PIL import Image


def _parse_rgb(raw: str) -> Tuple[int, int, int]:
    toks = [t.strip() for t in raw.split(",")]
    if len(toks) != 3:
        raise ValueError(f"invalid rgb string: {raw}")
    vals = [max(0, min(255, int(v))) for v in toks]
    return vals[0], vals[1], vals[2]


def _load_images(paths: List[str]) -> List[Image.Image]:
    out: List[Image.Image] = []
    for p in paths:
        if not os.path.isfile(p):
            continue
        out.append(Image.open(p).convert("RGB"))
    return out


def _resize_to_height(img: Image.Image, target_h: int) -> Image.Image:
    if img.height == target_h:
        return img
    ratio = float(target_h) / max(1.0, float(img.height))
    w = max(1, int(round(float(img.width) * ratio)))
    return img.resize((w, target_h), Image.BICUBIC)


def main() -> int:
    ap = argparse.ArgumentParser("make_contact_sheet")
    ap.add_argument("--images", nargs="+", required=True, help="input image paths")
    ap.add_argument("--out", required=True, help="output png path")
    ap.add_argument("--pad", type=int, default=8, help="padding in px")
    ap.add_argument("--bg", type=str, default="18,18,18", help="background color R,G,B")
    ap.add_argument("--target_height", type=int, default=0, help="0 means use max input height")
    args = ap.parse_args()

    imgs = _load_images(args.images)
    if not imgs:
        raise RuntimeError("no valid input images")

    target_h = int(args.target_height)
    if target_h <= 0:
        target_h = max(im.height for im in imgs)
    target_h = max(1, target_h)

    resized = [_resize_to_height(im, target_h) for im in imgs]
    pad = max(0, int(args.pad))
    bg = _parse_rgb(args.bg)

    total_w = sum(im.width for im in resized) + pad * (len(resized) + 1)
    total_h = target_h + pad * 2
    canvas = Image.new("RGB", (total_w, total_h), color=bg)

    x = pad
    for im in resized:
        canvas.paste(im, (x, pad))
        x += im.width + pad

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    canvas.save(args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
