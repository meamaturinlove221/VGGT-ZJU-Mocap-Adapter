import re
import os
import glob
import csv

SCENES_ROOT = r"F:\dataset_practice\work\vggt_scenes\CoreView_390"
LOG_DIR = os.path.join(SCENES_ROOT, "_logs")
OUT_DIR = os.path.join(SCENES_ROOT, "_audit")
os.makedirs(OUT_DIR, exist_ok=True)

LOG_GLOB = os.path.join(LOG_DIR, "frame_*.log")
OUT_CSV = os.path.join(OUT_DIR, "summary.csv")

re_num_images = re.compile(r"num_images:\s+(\d+)")
re_num_reg = re.compile(r"num_registered_images:\s+(\d+)")
re_num_pts = re.compile(r"num_points3D:\s+(\d+)")
re_init_cost = re.compile(r"Initial cost\s*:\s*([0-9.]+)")
re_final_cost = re.compile(r"Final cost\s*:\s*([0-9.]+)")
re_term = re.compile(r"Termination\s*:\s*(.+)")
re_ba_report = re.compile(r"Bundle adjustment report:")


def m1(rx, txt):
    m = rx.search(txt)
    return m.group(1).strip() if m else ""


rows = []
for path in sorted(glob.glob(LOG_GLOB)):
    frame = os.path.splitext(os.path.basename(path))[0]  # frame_000123
    scene_dir = os.path.join(SCENES_ROOT, frame)
    sparse_dir = os.path.join(scene_dir, "sparse")
    points3d_bin = os.path.join(sparse_dir, "points3D.bin")
    cameras_bin = os.path.join(sparse_dir, "cameras.bin")
    images_bin = os.path.join(sparse_dir, "images.bin")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    ok = ("Traceback" not in txt) and bool(re_ba_report.search(txt))

    rows.append({
        "frame": frame,
        "scene_dir": scene_dir,
        "log_path": path,
        "has_model_bin": int(os.path.exists(points3d_bin) and os.path.exists(cameras_bin) and os.path.exists(images_bin)),
        "num_images": m1(re_num_images, txt),
        "num_registered": m1(re_num_reg, txt),
        "num_points3D": m1(re_num_pts, txt),
        "init_cost_px": m1(re_init_cost, txt),
        "final_cost_px": m1(re_final_cost, txt),
        "termination": m1(re_term, txt),
        "ok": int(ok),
    })

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
    if rows:
        w.writeheader()
        w.writerows(rows)

print("wrote:", OUT_CSV, "rows=", len(rows))
