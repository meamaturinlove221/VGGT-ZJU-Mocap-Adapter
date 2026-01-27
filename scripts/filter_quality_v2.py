import os
import csv
import re

SCENES_ROOT = r'F:\dataset_practice\work\vggt_scenes\CoreView_390'
AUDIT_DIR = os.path.join(SCENES_ROOT, '_audit')
SUMMARY_CSV = os.path.join(AUDIT_DIR, 'summary.csv')

OUT_GOOD = os.path.join(AUDIT_DIR, 'good_frames.txt')
OUT_BAD = os.path.join(AUDIT_DIR, 'bad_frames.txt')

# 更稳：不强依赖 ok/has_model_bin 字段
MIN_REG = 2
MIN_PTS = 2000
MAX_COST = 2000.0   # 如果 cost 缺失/解析失败就忽略 cost


def to_int(x, default=0):
    try:
        return int(float(x))
    except:
        return default


def to_float(x, default=None):
    try:
        return float(x)
    except:
        return default


def has_model(frame_name: str) -> bool:
    sparse = os.path.join(SCENES_ROOT, frame_name, 'sparse')
    return os.path.exists(os.path.join(sparse, 'points3D.bin')) or os.path.exists(os.path.join(sparse, 'points3D.txt'))


good, bad = [], []
reasons = {'no_model': 0, 'low_reg': 0,
           'low_pts': 0, 'high_cost': 0, 'missing_frame': 0}

with open(SUMMARY_CSV, 'r', encoding='utf-8') as f:
    r = csv.DictReader(f)
    rows = list(r)

if not rows:
    print('summary has 0 data rows -> check _logs and summarize_runs.py glob')
    open(OUT_GOOD, 'w', encoding='utf-8').close()
    open(OUT_BAD, 'w', encoding='utf-8').close()
    raise SystemExit(0)

# 打印列名，避免“字段不匹配你却不知道”
print('summary columns:', ','.join(rows[0].keys()))

for row in rows:
    frame = row.get('frame', '').strip()
    if not frame:
        continue

    if not os.path.exists(os.path.join(SCENES_ROOT, frame)):
        bad.append(frame)
        reasons['missing_frame'] += 1
        continue

    if not has_model(frame):
        bad.append(frame)
        reasons['no_model'] += 1
        continue

    num_reg = to_int(
        row.get('num_registered', row.get('num_registered_images', '0')))
    num_pts = to_int(
        row.get('num_points3D', row.get('num_points3D_points', '0')))
    cost = to_float(row.get('final_cost_px', row.get('final_cost', '')))

    if num_reg < MIN_REG:
        bad.append(frame)
        reasons['low_reg'] += 1
        continue
    if num_pts < MIN_PTS:
        bad.append(frame)
        reasons['low_pts'] += 1
        continue
    if (cost is not None) and (cost > MAX_COST):
        bad.append(frame)
        reasons['high_cost'] += 1
        continue

    good.append(frame)

with open(OUT_GOOD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(good) + ('\n' if good else ''))
with open(OUT_BAD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(bad) + ('\n' if bad else ''))

print('good:', len(good), 'bad:', len(bad))
print('reasons:', reasons)
print('wrote:', OUT_GOOD)
