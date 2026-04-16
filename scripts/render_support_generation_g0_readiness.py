import argparse
import json
import os
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return ""


def get_font(size: int, prefer_zh: bool = False) -> ImageFont.FreeTypeFont:
    candidates: List[str] = []
    if prefer_zh:
        candidates.extend(
            [
                r"C:\Windows\Fonts\msyh.ttc",
                r"C:\Windows\Fonts\msyhbd.ttc",
                r"C:\Windows\Fonts\simhei.ttf",
                r"C:\Windows\Fonts\simsun.ttc",
            ]
        )
    candidates.extend(
        [
            r"C:\Windows\Fonts\consola.ttf",
            r"C:\Windows\Fonts\arial.ttf",
        ]
    )
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_text_png(lines: List[str], out_path: str, prefer_zh: bool = False) -> None:
    font = get_font(24 if prefer_zh else 20, prefer_zh=prefer_zh)
    line_h = 34 if prefer_zh else 30
    padding = 28
    width = 1800
    height = padding * 2 + max(1, len(lines)) * line_h
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=(0, 0, 0), font=font)
        y += line_h
    img.save(out_path)


def build_en(multiframe: Dict[str, Any], contract: Dict[str, Any]) -> str:
    groups = multiframe["groups"]
    orig = groups["orig6v400"]["aggregate"]
    iter25 = groups["iter25"]["aggregate"]
    lines = [
        "# G0 Readiness Brief",
        "",
        "## Current conclusion",
        "",
        "1. `S0` is already a clean generation-only probe.",
        "2. `S0` is worse than `T0-smoke`, so support generation itself is harming dynamic humans.",
        "3. Multi-frame diagnosis shows `bg_only` sharply reduces foreground `depth_conf` suppression.",
        "4. `G0` is still readiness-only until a real cloud run produces a matched candidate/result row.",
        "",
        "## Multi-frame evidence",
        "",
        "### orig6v400 (4 frames)",
        f"- `all`: `support_fg_mean={fmt(orig['all']['support_fg_mean_mean'])}`, `support_bg_mean={fmt(orig['all']['support_bg_mean_mean'])}`",
        f"- `all`: `depth_conf_delta_fg_mean={fmt(orig['all']['depth_conf_delta_fg_mean_mean'])}`, `depth_conf_delta_bg_mean={fmt(orig['all']['depth_conf_delta_bg_mean_mean'])}`",
        f"- `bg_only`: `depth_conf_delta_fg_mean={fmt(orig['bg_only']['depth_conf_delta_fg_mean_mean'])}`, `depth_conf_delta_bg_mean={fmt(orig['bg_only']['depth_conf_delta_bg_mean_mean'])}`",
        "",
        "### iter25 / pointhead_smoke (4 frames)",
        f"- `all`: `depth_conf_delta_fg_mean={fmt(iter25['all']['depth_conf_delta_fg_mean_mean'])}`, `depth_conf_delta_bg_mean={fmt(iter25['all']['depth_conf_delta_bg_mean_mean'])}`",
        f"- `bg_only`: `depth_conf_delta_fg_mean={fmt(iter25['bg_only']['depth_conf_delta_fg_mean_mean'])}`, `depth_conf_delta_bg_mean={fmt(iter25['bg_only']['depth_conf_delta_bg_mean_mean'])}`",
        "",
        "## Proposed next cloud baseline",
        "",
        "Use `G0` with:",
        f"- `pointmap_source = {contract.get('pointmap_source','')}`",
        f"- `point_target_mode = {contract.get('point_target_mode','')}`",
        f"- `precompute_mv_support_on = {contract.get('precompute_mv_support_on','')}`",
        f"- `precompute_mv_support_region_mode = {contract.get('precompute_mv_support_region_mode','')}`",
        f"- `point_support_mode = {contract.get('point_support_mode','')}`",
        f"- `point_mv_depth_support_mode = {contract.get('point_mv_depth_support_mode','')}`",
        f"- `point_mv_mask_support_mode = {contract.get('point_mv_mask_support_mode','')}`",
        f"- `point_target_blend_by_mv_support = {contract.get('point_target_blend_by_mv_support','')}`",
        "",
        "## Decision",
        "",
        "Do not resume cloud yet. Keep default near `T0-smoke`.",
        "If a new cloud slot is given, `G0` should be the first candidate.",
    ]
    return "\n".join(lines) + "\n"


def build_zh(multiframe: Dict[str, Any], contract: Dict[str, Any]) -> str:
    groups = multiframe["groups"]
    orig = groups["orig6v400"]["aggregate"]
    iter25 = groups["iter25"]["aggregate"]
    lines = [
        "# G0 就绪摘要",
        "",
        "## 当前结论",
        "",
        "1. `S0` 已经是干净的 generation-only probe。",
        "2. `S0` 比 `T0-smoke` 更差，说明 support 生成阶段本身就在伤动态人体。",
        "3. 多帧本地诊断表明，`bg_only` 会显著减轻前景 `depth_conf` 压制。",
        "4. `G0` 目前仍只是 readiness-only，只有真实云端运行产出匹配 candidate/result 后，才算实验结论。",
        "",
        "## 多帧证据",
        "",
        "### orig6v400（4 帧）",
        f"- `all`：`support_fg_mean={fmt(orig['all']['support_fg_mean_mean'])}`，`support_bg_mean={fmt(orig['all']['support_bg_mean_mean'])}`",
        f"- `all`：`depth_conf_delta_fg_mean={fmt(orig['all']['depth_conf_delta_fg_mean_mean'])}`，`depth_conf_delta_bg_mean={fmt(orig['all']['depth_conf_delta_bg_mean_mean'])}`",
        f"- `bg_only`：`depth_conf_delta_fg_mean={fmt(orig['bg_only']['depth_conf_delta_fg_mean_mean'])}`，`depth_conf_delta_bg_mean={fmt(orig['bg_only']['depth_conf_delta_bg_mean_mean'])}`",
        "",
        "### iter25 / pointhead_smoke（4 帧）",
        f"- `all`：`depth_conf_delta_fg_mean={fmt(iter25['all']['depth_conf_delta_fg_mean_mean'])}`，`depth_conf_delta_bg_mean={fmt(iter25['all']['depth_conf_delta_bg_mean_mean'])}`",
        f"- `bg_only`：`depth_conf_delta_fg_mean={fmt(iter25['bg_only']['depth_conf_delta_fg_mean_mean'])}`，`depth_conf_delta_bg_mean={fmt(iter25['bg_only']['depth_conf_delta_bg_mean_mean'])}`",
        "",
        "## 下一条最值得给云端窗口的基线",
        "",
        "使用 `G0`：",
        f"- `pointmap_source = {contract.get('pointmap_source','')}`",
        f"- `point_target_mode = {contract.get('point_target_mode','')}`",
        f"- `precompute_mv_support_on = {contract.get('precompute_mv_support_on','')}`",
        f"- `precompute_mv_support_region_mode = {contract.get('precompute_mv_support_region_mode','')}`",
        f"- `point_support_mode = {contract.get('point_support_mode','')}`",
        f"- `point_mv_depth_support_mode = {contract.get('point_mv_depth_support_mode','')}`",
        f"- `point_mv_mask_support_mode = {contract.get('point_mv_mask_support_mode','')}`",
        f"- `point_target_blend_by_mv_support = {contract.get('point_target_blend_by_mv_support','')}`",
        "",
        "## 当前决策",
        "",
        "现在不要恢复云端。当前默认继续贴近 `T0-smoke`。",
        "如果之后只给一个新的云端窗口，优先给 `G0`，不要先回去跑 `S1/S2/S3`。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--multiframe-json", required=True)
    ap.add_argument("--contract-json", required=True)
    ap.add_argument("--out-md-en", required=True)
    ap.add_argument("--out-png-en", required=True)
    ap.add_argument("--out-md-zh", required=True)
    ap.add_argument("--out-png-zh", required=True)
    args = ap.parse_args()

    multiframe = read_json(args.multiframe_json)
    contract = read_json(args.contract_json)
    text_en = build_en(multiframe, contract)
    text_zh = build_zh(multiframe, contract)

    for path in [args.out_md_en, args.out_png_en, args.out_md_zh, args.out_png_zh]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(args.out_md_en, "w", encoding="utf-8-sig") as f:
        f.write(text_en)
    with open(args.out_md_zh, "w", encoding="utf-8-sig") as f:
        f.write(text_zh)

    render_text_png(text_en.splitlines(), args.out_png_en, prefer_zh=False)
    render_text_png(text_zh.splitlines(), args.out_png_zh, prefer_zh=True)
    print(f"[render-g0-readiness] wrote {args.out_md_en} {args.out_md_zh}")


if __name__ == "__main__":
    main()
