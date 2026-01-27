import os
import csv

SCENES_ROOT = r"F:\dataset_practice\work\vggt_scenes\CoreView_390"
AUDIT_DIR = os.path.join(SCENES_ROOT, "_audit")
SUMMARY_CSV = os.path.join(AUDIT_DIR, "summary.csv")

OUT_GOOD = os.path.join(AUDIT_DIR, "good_frames.txt")
OUT_BAD = os.path.join(AUDIT_DIR, "bad_frames.txt")

# 宽松阈值：先保证能筛出一批“看起来正常”的
GOOD_MIN_REG = 6
GOOD_MIN_PTS = 1500
GOOD_MAX_COST = 500.0


def to_int(x, default=0):
    try:
        return int(float(x))
    except:
        return default


def to_float(x, default=1e18):
    try:
        return float(x)
    except:
        return default


good, bad = [], []
with open(SUMMARY_CSV, "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        frame = row["frame"]
        num_reg = to_int(row.get("num_registered", "0"))
        num_pts = to_int(row.get("num_points3D", "0"))
        cost = to_float(row.get("final_cost_px", ""))
        ok = to_int(row.get("ok", "0"))
        has_bin = to_int(row.get("has_model_bin", "0"))

        is_good = (ok == 1) and (has_bin == 1) and (num_reg >= GOOD_MIN_REG) and (
            num_pts >= GOOD_MIN_PTS) and (cost <= GOOD_MAX_COST)
        (good if is_good else bad).append(frame)

with open(OUT_GOOD, "w", encoding="utf-8") as f:
    f.write("\n".join(good) + ("\n" if good else ""))
with open(OUT_BAD, "w", encoding="utf-8") as f:
    f.write("\n".join(bad) + ("\n" if bad else ""))

print("good:", len(good), "bad:", len(bad))
print("wrote:", OUT_GOOD)
