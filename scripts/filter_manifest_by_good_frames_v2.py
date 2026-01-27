import argparse
import json
import re
from pathlib import Path

re_frame = re.compile(r'frame_\d{6}')


def extract_frame(obj, raw_line: str):
    if isinstance(obj, dict):
        v = obj.get('frame', None)
        if isinstance(v, str) and re_frame.search(v):
            return re_frame.search(v).group(0)
        v = obj.get('scene_dir', None)
        if isinstance(v, str) and re_frame.search(v):
            return re_frame.search(v).group(0)
    m = re_frame.search(raw_line)
    return m.group(0) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_jsonl', required=True)
    ap.add_argument('--good_frames', required=True)
    ap.add_argument('--out_jsonl', required=True)
    args = ap.parse_args()

    good = set([x.strip() for x in Path(args.good_frames).read_text(
        encoding='utf-8', errors='ignore').splitlines() if x.strip()])
    inp = Path(args.in_jsonl)
    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0
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
            fr = extract_frame(obj, line)
            if fr and fr in good:
                f_out.write(line + '\n')
                kept += 1

    print('total lines:', total)
    print('good frames:', len(good))
    print('kept lines :', kept)
    print('wrote:', str(outp))


if __name__ == '__main__':
    main()
