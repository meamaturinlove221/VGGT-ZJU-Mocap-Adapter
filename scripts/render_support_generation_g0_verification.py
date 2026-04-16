import argparse
import json
import os
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def to_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def fmt(value: Any, digits: int = 4) -> str:
    num = to_float(value)
    if num is None:
        return ""
    return f"{num:.{digits}f}"


def delta(a: Dict[str, Any], b: Dict[str, Any], key: str) -> str:
    va = to_float(a.get(key))
    vb = to_float(b.get(key))
    if va is None or vb is None:
        return ""
    return f"{va - vb:+.4f}"


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
    candidates.extend([r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\arial.ttf"])
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


def build_md(t0: Dict[str, Any], s0: Dict[str, Any], g0: Dict[str, Any], zh: bool) -> str:
    if zh:
        lines = [
            "# G0 单次验证摘要",
            "",
            "## 结论",
            "",
            "1. `G0` 已经跑过真实云端单窗口，不再是 readiness-only。",
            "2. `G0` 相比 `S0` 明显更好，说明 `support generation bg_only` 能缓解 support 生成阶段对动态人体的伤害。",
            "3. `G0` 相比 `T0-smoke`，主 ghost 指标更好，但前景存在感指标仍略弱，所以它更像“更安全的 support-enabled 候选”，还不是已经可以替代当前默认的终局方案。",
            "",
            "## 核心指标",
            "",
            "| Probe | ghost_visual | fg_luma | fg_contrast | fg_tgt_l1 |",
            "|---|---|---|---|---|",
            f"| T0-smoke | {fmt(t0.get('ghost_visual_score'))} | {fmt(t0.get('fg_pred_luma_mean'))} | {fmt(t0.get('fg_pred_contrast'))} | {fmt(t0.get('fg_pred_tgt_l1'))} |",
            f"| S0 | {fmt(s0.get('ghost_visual_score'))} | {fmt(s0.get('fg_pred_luma_mean'))} | {fmt(s0.get('fg_pred_contrast'))} | {fmt(s0.get('fg_pred_tgt_l1'))} |",
            f"| G0 | {fmt(g0.get('ghost_visual_score'))} | {fmt(g0.get('fg_pred_luma_mean'))} | {fmt(g0.get('fg_pred_contrast'))} | {fmt(g0.get('fg_pred_tgt_l1'))} |",
            "",
            "## G0 相对对照",
            "",
            f"- G0 vs S0: `ghost_visual={delta(g0, s0, 'ghost_visual_score')}`，`fg_luma={delta(g0, s0, 'fg_pred_luma_mean')}`，`fg_contrast={delta(g0, s0, 'fg_pred_contrast')}`，`fg_tgt_l1={delta(g0, s0, 'fg_pred_tgt_l1')}`",
            f"- G0 vs T0-smoke: `ghost_visual={delta(g0, t0, 'ghost_visual_score')}`，`fg_luma={delta(g0, t0, 'fg_pred_luma_mean')}`，`fg_contrast={delta(g0, t0, 'fg_pred_contrast')}`，`fg_tgt_l1={delta(g0, t0, 'fg_pred_tgt_l1')}`",
            "",
            "## 当前判断",
            "",
            "当前最合理的解读是：`bg_only` support generation 是正确方向，但它更像“止血”而不是终局。",
            "如果之后再给 support 一个云端窗口，优先继续从 `G0` 出发，而不是回退到 `S1/S2/S3` 或旧的 all-support 基线。",
        ]
    else:
        lines = [
            "# G0 Single-Window Verification",
            "",
            "## Conclusion",
            "",
            "1. `G0` is now a real cloud result, not readiness-only.",
            "2. `G0` is clearly better than `S0`, so `support generation bg_only` mitigates the generation-stage harm on dynamic humans.",
            "3. `G0` is better than `T0-smoke` on the primary ghost metric, but still slightly weaker on some foreground-presence metrics. It should be treated as a safer support-enabled candidate, not yet a final replacement for the current default.",
            "",
            "## Core Metrics",
            "",
            "| Probe | ghost_visual | fg_luma | fg_contrast | fg_tgt_l1 |",
            "|---|---|---|---|---|",
            f"| T0-smoke | {fmt(t0.get('ghost_visual_score'))} | {fmt(t0.get('fg_pred_luma_mean'))} | {fmt(t0.get('fg_pred_contrast'))} | {fmt(t0.get('fg_pred_tgt_l1'))} |",
            f"| S0 | {fmt(s0.get('ghost_visual_score'))} | {fmt(s0.get('fg_pred_luma_mean'))} | {fmt(s0.get('fg_pred_contrast'))} | {fmt(s0.get('fg_pred_tgt_l1'))} |",
            f"| G0 | {fmt(g0.get('ghost_visual_score'))} | {fmt(g0.get('fg_pred_luma_mean'))} | {fmt(g0.get('fg_pred_contrast'))} | {fmt(g0.get('fg_pred_tgt_l1'))} |",
            "",
            "## G0 Delta",
            "",
            f"- G0 vs S0: `ghost_visual={delta(g0, s0, 'ghost_visual_score')}`, `fg_luma={delta(g0, s0, 'fg_pred_luma_mean')}`, `fg_contrast={delta(g0, s0, 'fg_pred_contrast')}`, `fg_tgt_l1={delta(g0, s0, 'fg_pred_tgt_l1')}`",
            f"- G0 vs T0-smoke: `ghost_visual={delta(g0, t0, 'ghost_visual_score')}`, `fg_luma={delta(g0, t0, 'fg_pred_luma_mean')}`, `fg_contrast={delta(g0, t0, 'fg_pred_contrast')}`, `fg_tgt_l1={delta(g0, t0, 'fg_pred_tgt_l1')}`",
            "",
            "## Current Reading",
            "",
            "`bg_only` support generation looks like the correct direction, but still more like mitigation than end-state.",
            "If another support-enabled window is granted later, continue from `G0` rather than going back to old all-support baselines.",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0-json", required=True)
    ap.add_argument("--s0-json", required=True)
    ap.add_argument("--g0-json", required=True)
    ap.add_argument("--out-md-en", required=True)
    ap.add_argument("--out-png-en", required=True)
    ap.add_argument("--out-md-zh", required=True)
    ap.add_argument("--out-png-zh", required=True)
    args = ap.parse_args()

    t0 = read_json(args.t0_json)
    s0 = read_json(args.s0_json)
    g0 = read_json(args.g0_json)
    text_en = build_md(t0, s0, g0, zh=False)
    text_zh = build_md(t0, s0, g0, zh=True)

    for path in [args.out_md_en, args.out_png_en, args.out_md_zh, args.out_png_zh]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(args.out_md_en, "w", encoding="utf-8-sig") as f:
        f.write(text_en)
    with open(args.out_md_zh, "w", encoding="utf-8-sig") as f:
        f.write(text_zh)
    render_text_png(text_en.splitlines(), args.out_png_en, prefer_zh=False)
    render_text_png(text_zh.splitlines(), args.out_png_zh, prefer_zh=True)
    print(f"[render-g0-verification] wrote {args.out_md_en} {args.out_md_zh}")


if __name__ == "__main__":
    main()
