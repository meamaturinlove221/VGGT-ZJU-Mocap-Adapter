import argparse
import json
import re
from pathlib import Path

RE_FRAME6 = re.compile(r'(?<!\d)(\d{6})(?!\d)')
RE_IMG6 = re.compile(r'[\\/](\d{6})\.(png|jpg|jpeg)', re.IGNORECASE)

FRAME_KEYS = ['frame', 'frame_id', 'idx', 'index', 't', 'time', 'fid']
PATH_KEYS = ['img', 'image', 'image_path', 'rgb', 'rgb_path',
             'path', 'file', 'mask', 'mask_path', 'fg_mask', 'valid_mask']


def load_good(good_frames_path: str):
    lines = Path(good_frames_path).read_text(
        encoding='utf-8', errors='ignore').splitlines()
    good_int = set()
    good_str = set()
    for s in lines:
        s = s.strip()
        if not s:
            continue
        m = re.search(r'(\d{6})', s)
        if m:
            good_str.add(m.group(1))
            good_int.add(int(m.group(1)))
        else:
            # 也允许纯数字 0..n
            try:
                good_int.add(int(s))
                good_str.add(f'{int(s):06d}')
            except:
                pass
    return good_int, good_str


def extract_frame_id(obj: dict):
    # 1) 直接字段
    for k in FRAME_KEYS:
        if k in obj:
            v = obj[k]
            if isinstance(v, int):
                return v, f'{v:06d}'
            if isinstance(v, str):
                v2 = v.strip()
                if v2.isdigit():
                    i = int(v2)
                    return i, f'{i:06d}'
                m = RE_FRAME6.search(v2)
                if m:
                    i = int(m.group(1))
                    return i, m.group(1)
    # 2) 从 path 字段里抓 000000.png / 000000.jpg
    for k in PATH_KEYS:
        if k in obj and isinstance(obj[k], str):
            m = RE_IMG6.search(obj[k])
            if m:
                i = int(m.group(1))
                return i, m.group(1)
            m = RE_FRAME6.search(obj[k])
            if m:
                i = int(m.group(1))
                return i, m.group(1)
    # 3) 从所有字符串值里兜底抓
    for v in obj.values():
        if isinstance(v, str):
            m = RE_IMG6.search(v)
            if m:
                i = int(m.group(1))
                return i, m.group(1)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_jsonl', required=True)
    ap.add_argument('--good_frames', required=True)
    ap.add_argument('--out_jsonl', required=True)
    args = ap.parse_args()

    good_int, good_str = load_good(args.good_frames)
    inp = Path(args.in_jsonl)
    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0
    miss = 0

    with inp.open('r', encoding='utf-8', errors='ignore') as f_in, outp.open('w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue

            i, s6 = extract_frame_id(obj)
            if i is None:
                miss += 1
                continue

            if (i in good_int) or (s6 in good_str):
                f_out.write(line + '\n')
                kept += 1

    print('total lines:', total)
    print('good frames:', len(good_int))
    print('kept lines :', kept)
    print('unparsed frame lines:', miss)
    print('wrote:', str(outp))


if __name__ == '__main__':
    main()
