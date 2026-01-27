# make_overlays.py
import json
import random
from pathlib import Path
from PIL import Image
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True,
                    help="pack root, contains images/ and masks/")
    ap.add_argument("--good", required=True, help="good jsonl path")
    ap.add_argument("--out", default=None,
                    help="output dir, default: <pack>/qc_overlay")
    ap.add_argument("--num", type=int, default=20, help="number of samples")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--alpha", type=int, default=120,
                    help="mask overlay alpha 0-255")
    args = ap.parse_args()

    pack = Path(args.pack)
    src = Path(args.good)
    outd = Path(args.out) if args.out else (pack / "qc_overlay")
    outd.mkdir(parents=True, exist_ok=True)

    # read lines
    lines = []
    with src.open("r", encoding="utf-8", errors="ignore") as f:
        for l in f:
            l = l.strip()
            if l:
                lines.append(l)

    random.seed(args.seed)
    k = min(args.num, len(lines))
    picked = random.sample(lines, k) if k > 0 else []

    rec = []
    for l in picked:
        try:
            rec.append(json.loads(l))
        except Exception:
            continue

    html = []
    n_written = 0

    for o in rec:
        fr = str(o.get("frame", ""))
        cam = str(o.get("cam", ""))
        if not fr or not cam:
            continue

        # prefer jpg, fallback png
        ip = pack / "images" / cam / (fr + ".jpg")
        if not ip.exists():
            ip = pack / "images" / cam / (fr + ".png")

        mp = pack / "masks" / cam / (fr + ".png")

        if not ip.exists() or not mp.exists():
            # still record in html with a note
            html.append(
                f"<div style='margin:12px'>"
                f"<div><b>{fr}</b> {cam} <span style='color:#b00'>(missing file)</span></div>"
                f"<div>img={ip} exists={ip.exists()}<br>msk={mp} exists={mp.exists()}</div>"
                f"</div>"
            )
            continue

        try:
            img = Image.open(ip).convert("RGB")
            m = Image.open(mp).convert("L")
        except Exception as e:
            html.append(
                f"<div style='margin:12px'>"
                f"<div><b>{fr}</b> {cam} <span style='color:#b00'>(open failed)</span></div>"
                f"<div>{e}</div></div>"
            )
            continue

        if m.size != img.size:
            m = m.resize(img.size, Image.NEAREST)

        a = m.point(lambda x: args.alpha if x > 0 else 0)
        red = Image.new("L", img.size, 255)
        zero = Image.new("L", img.size, 0)
        rgba = Image.merge("RGBA", (red, zero, zero, a))
        out = Image.alpha_composite(img.convert("RGBA"), rgba).convert("RGB")

        fn = f"{fr}_{cam}.png"
        out.save(outd / fn)
        n_written += 1

        html.append(
            f"<div style='margin:12px'>"
            f"<div><b>{fr}</b> {cam}</div>"
            f"<img src='{fn}' style='max-width:1100px'>"
            f"</div>"
        )

    (outd / "index.html").write_text(
        "<html><body>" + "".join(html) + "</body></html>",
        encoding="utf-8"
    )

    print("wrote", n_written, "overlays ->", outd)


if __name__ == "__main__":
    main()
