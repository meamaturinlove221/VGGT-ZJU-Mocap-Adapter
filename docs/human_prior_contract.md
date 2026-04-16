# Human Prior Sidecar Contract

`finetune_vggt_pseudo.py` 现在支持可选的人体先验 sidecar，不改 VGGT backbone，直接在伪监督训练阶段叠加额外监督。

## 放置路径

默认配置:

```text
<zju_root>/<seq>/<human_prior_subdir>/<geom_npz_basename>.npz
```

例子:

```text
<zju_root>/0012_11/human_prior/frame_000123.npz
<zju_root>/0012_11/vggt_geom/frame_000123.npz
```

其中 sidecar 文件名要和几何缓存 `.npz` 同名。

如果 `--human_prior_subdir` 给的是绝对路径，还会额外尝试:

```text
<abs_subdir>/<seq>/<geom_npz_basename>.npz
<abs_subdir>/<geom_npz_basename>.npz
```

## 支持的键

点图:

- `prior_pointmap`
- `smpl_pointmap`
- `smplx_pointmap`
- `pointmap`

有效区域:

- `prior_valid_mask`
- `valid_mask`
- `smpl_valid_mask`

身体区域:

- `body_mask`
- `prior_body_mask`
- `smpl_body_mask`
- `mask`

头部区域:

- `head_mask`
- `prior_head_mask`
- `smpl_head_mask`

脸部区域:

- `face_mask`
- `prior_face_mask`
- `smpl_face_mask`

元信息:

- `cam_names`
- `pointmap_frame` 或 `prior_pointmap_frame`
- `prior_source` 或 `source` 或 `prior_type`

## 张量形状

- `pointmap`: `(V, H, W, 3)`
- `valid_mask/body_mask/head_mask/face_mask`: `(V, H, W)` 为最佳
- mask 也接受 `(H, W)` 或 `(V, H, W, 1/3/4)`，训练时会归一成 `(V, H, W)`

说明:

- `V` 应与当前样本视图数一致
- 如果 sidecar 内提供了 `cam_names`，会按相机名对齐
- 如果没有 `cam_names`，则按视图顺序对齐
- 若 sidecar 分辨率和训练监督分辨率不同，训练时会自动 resize 到当前监督分辨率

## `pointmap_frame`

推荐明确写成:

- `world`
- `camera`

如果留空或写成其他值，训练代码会尝试自动判断；能明确写就不要依赖自动判断。

## 区域策略

以下参数都支持这些区域模式:

- `off`
- `all`
- `body`
- `head`
- `face`
- `head_face`

如果没有显式 `head_mask` / `face_mask`:

- `head` 会从 `body` 或 `valid` 的顶部条带回退生成
- `face` 会从 `head` 或 `body` 的顶部条带继续回退生成

这样可以先用粗人体先验开跑，不必等到精细脸部标签齐全。

## 训练开关

最小示例:

```bash
python finetune_vggt_pseudo.py ^
  --human_prior_enable on ^
  --human_prior_subdir human_prior ^
  --human_prior_point_blend_alpha 0.35 ^
  --human_prior_point_blend_region head_face ^
  --human_prior_weight_boost 1.5 ^
  --human_prior_weight_region body ^
  --human_prior_complete_weight 0.35 ^
  --human_prior_complete_region body ^
  --lambda_point_prior 0.15 ^
  --human_prior_loss_region head_face
```

建议起步:

- `--human_prior_point_blend_alpha 0.2~0.4`
- `--human_prior_weight_boost 1.2~1.8`
- `--human_prior_complete_weight 0.2~0.5`
- `--lambda_point_prior 0.05~0.2`

## 行为说明

- `human_prior_point_blend_alpha`: 把 prior 点图按区域混入当前 `point_tgt_for_loss`
- `human_prior_weight_boost`: 在指定区域提高监督权重
- `human_prior_complete_weight`: 在伪几何缺失、但 prior 有效的区域补一条 point supervision 覆盖，用来补洞
- `lambda_point_prior`: 对 prior 点图额外加一项 point loss
- hair 不建议一开始就强约束，可先把 blend/loss 区域设为 `head_face` 或 `body`

## 完整性建议

如果目标是先把人体点云补完整，而不是先抠极细头发:

- `human_prior_complete_region=body`
- `human_prior_complete_weight=0.25~0.5`
- `human_prior_point_blend_region=head_face`
- `human_prior_loss_region=head_face` 或 `body`

这样会优先让 prior 去填补伪几何没覆盖到的身体/头部区域，而不是只在已有监督上做加权。

## 严格模式

打开 `--human_prior_strict on` 后:

- sidecar 缺失会直接报错
- `cam_names` 对不上会直接报错
- 视图数不一致会直接报错

适合正式批量上云前做数据体检。
