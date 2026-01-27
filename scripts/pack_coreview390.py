# F:\vggt\scripts\pack_coreview390.py
import os
import json
import random
import csv
import shutil
import re
from pathlib import Path

# === 只改这里两行 ===
GOOD = r"F:\dataset_practice\manifests\coreview390_with_mask.good.jsonl"
PACK = r"F:\dataset_practice\work\coreview390_good_pack"
# ====================


def _norm_frame(x: str) -> str:
    s = ("" if x is None else str(x)).strip()
    if s.lower().startswith("frame_"):
        s = s[6:]
    m = re.search(r"(\d+)", s)
    if m:
        s = m.group(1)
    return s.zfill(6) if s.isdigit() else s


def _load_frameset(p: Path) -> set:
    if not p.exists():
        return set()
    txt = p.read_text(encoding="utf-8", errors="ignore")
    # 把字面量 \n 变成真实换行 + 兼容各种换行
    txt = txt.replace("\r\n", "\n").replace("\r", "\n").replace("\\n", "\n")
    items = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        # 这一行可能还包含空格分隔的多个 token
        for tok in line.split():
            tok = tok.strip()
            if tok:
                items.append(_norm_frame(tok))
    # 写回标准多行文件
    p.write_text("\n".join(items) + "\n", encoding="ascii")
    return set(items)


def _ensure_splits(meta: Path, frames_all: list[str]):
    tf, vf, ef = meta/"train_frames.txt", meta / \
        "val_frames.txt", meta/"test_frames.txt"
    tr = _load_frameset(tf)
    va = _load_frameset(vf)
    te = _load_frameset(ef)
    if tr and va and te:
        return tr, va, te

    # 如果 split 文件不存在/空，自动按 0.8/0.1/0.1 切（20 帧会是 16/2/2）
    frames = list(frames_all)
    random.seed(42)
    random.shuffle(frames)
    n = len(frames)
    n_tr = max(1, int(n * 0.8))
    n_va = max(1, int(n * 0.1))
    n_te = max(1, n - n_tr - n_va)
    tr_l = frames[:n_tr]
    va_l = frames[n_tr:n_tr+n_va]
    te_l = frames[n_tr+n_va:n_tr+n_va+n_te]
    tf.write_text("\n".join(tr_l) + "\n", encoding="ascii")
    vf.write_text("\n".join(va_l) + "\n", encoding="ascii")
    ef.write_text("\n".join(te_l) + "\n", encoding="ascii")
    return set(tr_l), set(va_l), set(te_l)


def _split_good_jsonl(good_path: Path, meta: Path, tr: set, va: set, te: set):
    outs = {
        "train": (meta/"train.jsonl").open("w", encoding="utf-8", newline="\n"),
        "val": (meta/"val.jsonl").open("w", encoding="utf-8", newline="\n"),
        "test": (meta/"test.jsonl").open("w", encoding="utf-8", newline="\n"),
    }
    n = {"train": 0, "val": 0, "test": 0}
    miss = 0
    with good_path.open("r", encoding="utf-8", errors="ignore") as f:
        for l in f:
            if not l.strip():
                continue
            s = l.lstrip("\ufeff").rstrip("\r\n")
            try:
                o = json.loads(s)
            except Exception:
                continue
            fr = _norm_frame(o.get("frame", ""))
            k = "train" if fr in tr else (
                "val" if fr in va else ("test" if fr in te else None))
            if k is None:
                miss += 1
                continue
            outs[k].write(s + "\n")
            n[k] += 1
    for fp in outs.values():
        fp.close()
    return n, miss


def _viz_one_split(pack: Path, meta: Path, split: str, sample_n: int):
    try:
        from PIL import Image
        import numpy as np
    except Exception:
        raise RuntimeError("需要安装 pillow 和 numpy：pip install pillow numpy")

    src = meta/(split + ".jsonl")
    out = pack/"viz"/split
    out.mkdir(parents=True, exist_ok=True)

    lines = [l for l in src.read_text(
        encoding="utf-8", errors="ignore").splitlines() if l.strip()]
    random.seed(42 + {"train": 1, "val": 2, "test": 3}[split])
    random.shuffle(lines)
    lines = lines[:min(sample_n, len(lines))]

    wrote, miss = 0, 0
    for i, l in enumerate(lines):
        try:
            o = json.loads(l)
        except Exception:
            miss += 1
            continue
        ip = o.get("image_path")
        mp = o.get("mask_path")
        cam = str(o.get("cam", ""))
        fr = _norm_frame(o.get("frame", ""))
        if (not ip) or (not mp):
            miss += 1
            continue
        try:
            img = Image.open(ip).convert("RGB")
            m = Image.open(mp).convert("L")
        except Exception:
            miss += 1
            continue
        if m.size != img.size:
            m = m.resize(img.size, resample=Image.NEAREST)

        mb = (np.array(m) > 0)
        stem = (cam + "_" + fr) if cam and fr else f"{i:04d}"

        img.save(out/(stem+"_rgb.jpg"), quality=95)
        Image.fromarray((mb.astype("uint8")*255)).save(out/(stem+"_mask.png"))

        arr = np.array(img).astype("float32")
        color = np.array([255, 0, 0], dtype="float32")
        alpha = 0.45
        arr[mb] = arr[mb]*(1-alpha) + color*alpha
        Image.fromarray(arr.clip(0, 255).astype("uint8")).save(
            out/(stem+"_overlay.jpg"), quality=95)
        wrote += 1
    return wrote, miss, out


def _write_index_html(pack: Path):
    import html
    viz = pack/"viz"
    idx = viz/"index.html"
    parts = [
        "<!doctype html><meta charset=utf-8><title>coreview390 audit</title>",
        "<style>body{font-family:Arial;margin:18px} .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px} img{max-width:100%;height:auto;border:1px solid #ddd} .card{padding:8px;border:1px solid #eee;border-radius:10px}</style>",
        "<h1>CoreView_390 good-pack audit</h1>",
    ]
    for sp in ("train", "val", "test"):
        d = viz/sp
        if not d.exists():
            continue
        parts.append(f"<h2>{sp}</h2><div class=grid>")
        for ov in sorted(d.glob("*_overlay.jpg")):
            stem = ov.name[:-len("_overlay.jpg")]
            rgb = d/(stem+"_rgb.jpg")
            ms = d/(stem+"_mask.png")
            parts.append(
                "<div class=card><div>"+html.escape(stem)+"</div>" +
                (f"<div><b>RGB</b><br><a href='{sp}/{rgb.name}'><img src='{sp}/{rgb.name}'></a></div>" if rgb.exists() else "") +
                f"<div><b>OVERLAY</b><br><a href='{sp}/{ov.name}'><img src='{sp}/{ov.name}'></a></div>" +
                (f"<div><b>MASK</b><br><a href='{sp}/{ms.name}'><img src='{sp}/{ms.name}'></a></div>" if ms.exists() else "") +
                "</div>"
            )
        parts.append("</div>")
    idx.write_text("\n".join(parts), encoding="utf-8")
    return idx


def _make_portable(pack: Path, meta: Path):
    outroot = pack/"data"
    (outroot/"images").mkdir(parents=True, exist_ok=True)
    (outroot/"masks").mkdir(parents=True, exist_ok=True)

    rel_outs = {}
    for sp in ("train", "val", "test"):
        src = meta/(sp+".jsonl")
        dst = meta/(sp+"_rel.jsonl")
        rel_outs[sp] = (src, dst)

    counts = {}
    for sp, (src, dst) in rel_outs.items():
        cnt = 0
        with src.open("r", encoding="utf-8", errors="ignore") as f, dst.open("w", encoding="utf-8", newline="\n") as g:
            for l in f:
                if not l.strip():
                    continue
                o = json.loads(l)
                cam = str(o.get("cam", "")).strip()
                fr = _norm_frame(o.get("frame", ""))
                ip = o.get("image_path")
                mp = o.get("mask_path")
                if (not cam) or (not fr) or (not ip) or (not mp):
                    continue
                rel_i = Path("images")/cam/(fr+".jpg")
                rel_m = Path("masks")/cam/(fr+".png")
                (outroot/rel_i).parent.mkdir(parents=True, exist_ok=True)
                (outroot/rel_m).parent.mkdir(parents=True, exist_ok=True)
                if not (outroot/rel_i).exists():
                    shutil.copy2(ip, outroot/rel_i)
                if not (outroot/rel_m).exists():
                    shutil.copy2(mp, outroot/rel_m)
                o["image_path"] = str(rel_i).replace("\\", "/")
                o["mask_path"] = str(rel_m).replace("\\", "/")
                g.write(json.dumps(o, ensure_ascii=False) + "\n")
                cnt += 1
        counts[sp] = cnt
    return outroot, counts


def _split_stats(pack: Path, meta: Path, out_csv: Path):
    try:
        from PIL import Image
        import numpy as np
    except Exception:
        raise RuntimeError("需要安装 pillow 和 numpy：pip install pillow numpy")

    rows = []
    data = pack/"data"
    for sp in ("train", "val", "test"):
        p = meta/(sp+"_rel.jsonl")
        if not p.exists():
            continue
        agg = {}  # cam -> (cnt, sum_ratio)
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for l in f:
                if not l.strip():
                    continue
                o = json.loads(l)
                cam = str(o.get("cam", "")).strip()
                fr = _norm_frame(o.get("frame", ""))
                mp = o.get("mask_path")
                if (not cam) or (not fr) or (not mp):
                    continue
                mp_abs = data / Path(mp)
                if not mp_abs.exists():
                    continue
                m = np.array(Image.open(mp_abs).convert("L"))
                ratio = float((m > 0).mean())
                c, s = agg.get(cam, (0, 0.0))
                agg[cam] = (c+1, s+ratio)
        for cam, (c, s) in sorted(agg.items()):
            rows.append({"split": sp, "cam": cam, "count": c,
                        "mean_mask_ratio": (s/c if c else 0.0)})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["split", "cam", "count", "mean_mask_ratio"])
        w.writeheader()
        w.writerows(rows)
    return out_csv, len(rows)


def main():
    good = Path(GOOD)
    pack = Path(PACK)
    meta = pack/"meta"
    meta.mkdir(parents=True, exist_ok=True)
    pack.mkdir(parents=True, exist_ok=True)

    # 1) 收集全体帧号
    frames_all = sorted({_norm_frame(json.loads(l).get("frame", "")) for l in good.read_text(
        encoding="utf-8", errors="ignore").splitlines() if l.strip()})
    tr, va, te = _ensure_splits(meta, frames_all)

    # 2) 生成 train/val/test.jsonl
    n, miss = _split_good_jsonl(good, meta, tr, va, te)
    print("[split] wrote", n, "miss", miss, "->", meta)

    # 3) 可视化抽检
    w1, m1, p1 = _viz_one_split(pack, meta, "train", sample_n=80)
    w2, m2, p2 = _viz_one_split(pack, meta, "val",   sample_n=10**9)
    w3, m3, p3 = _viz_one_split(pack, meta, "test",  sample_n=10**9)
    print("[viz] train", w1, "miss", m1, "->", p1)
    print("[viz] val  ", w2, "miss", m2, "->", p2)
    print("[viz] test ", w3, "miss", m3, "->", p3)

    idx = _write_index_html(pack)
    print("[html] wrote", idx)

    # 4) 可移植数据包
    outroot, counts = _make_portable(pack, meta)
    print("[portable] root", outroot, "copied_lines", counts)

    # 5) split 统计
    out_csv, rows = _split_stats(pack, meta, meta/"split_stats.csv")
    print("[stats] wrote", out_csv, "rows", rows)


if __name__ == "__main__":
    main()
