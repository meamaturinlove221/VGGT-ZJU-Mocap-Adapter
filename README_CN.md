# VGGT-ZJU-MoCap Adapter

<p align="center">
  <b>面向 ZJU-MoCap 风格数据的 VGGT 人体先验实验数据桥。</b>
</p>

<p align="center">
  多视角 RGB · 相机绑定 · mask 审计 · SMPL/SMPL-X 先验对齐 · VGGT-ready case 导出
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="#这个仓库解决什么问题">解决什么问题</a> ·
  <a href="#处理流程">处理流程</a> ·
  <a href="#证据边界">证据边界</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/VGGT-human--prior-blue" alt="VGGT human prior" />
  <img src="https://img.shields.io/badge/ZJU--MoCap-dataset--adapter-0f766e" alt="ZJU-MoCap dataset adapter" />
  <img src="https://img.shields.io/badge/status-research--utility-orange" alt="research utility" />
  <img src="https://img.shields.io/badge/policy-failure--closed-red" alt="failure closed" />
</p>

<p align="center">
  <img src="docs/figures/vggt_zju_mocap_adapter_architecture.svg" alt="VGGT-ZJU-MoCap Adapter architecture" width="100%" />
</p>

---

## 这个仓库是什么

`VGGT-ZJU-MoCap-Adapter` 是 VGGT + 人体先验项目里的数据可靠性层。

它不负责包装最终结果，也不把诊断图当成果。这个仓库只做一件事：把 ZJU-MoCap 风格的数据整理成可以交给 VGGT 或人体先验分支使用的可信 case，并把相机、帧号、mask、人体先验投影这些关键环节查清楚。

后续模型到底有没有提升，必须建立在一个干净的数据入口上。这个仓库处理的就是这个入口。

---

## 这个仓库解决什么问题

多视角人体重建里，很多看起来像模型问题的现象，最后查下来其实是数据绑定问题。

常见情况包括：

- RGB 帧和相机文件对不上；
- 图片 resize 之后，内参没有同步更新；
- world-to-camera 和 camera-to-world 混着用；
- mask 边界不稳定，影响人体区域和背景保留；
- SMPL / SMPL-X 先验投影到图像上发生偏移；
- teacher/reference 产物被误当成 student 模型输出；
- projection overlay 看起来对了，但 3D 点云并没有真正成形。

这个仓库把这些问题提前暴露出来。数据层如果不可信，就应该在这里停止，而不是继续训练、继续截图、继续把问题往后推。

---

## 项目位置

```text
ZJU-MoCap-style data
        │
        ▼
VGGT-ZJU-MoCap-Adapter
  ├─ 路径与帧号统一
  ├─ 相机参数审计
  ├─ mask 审计
  ├─ 人体先验投影检查
  ├─ 诊断图与对照包
  └─ VGGT-ready case package
        │
        ├─ vanilla VGGT baseline
        │
        └─ VGGT-SMPL-X Human Prior Adapter
              │
              ▼
      human-main full-scene RGB point-cloud evidence
```

和其他仓库的关系：

- `VGGT-SMPL-X-Human-Prior-Adapter`：模型侧人体先验注入与训练路线。
- `vggt_for_4k_4d`：full-scene 证据、对照结果和报告整理路线。

这个仓库处在更前面，负责把 ZJU-MoCap 风格的数据先处理成可以讨论的输入。

---

## 处理流程

### 1. 收集 case

输入通常包含多视角 RGB、相机参数、mask、subject / sequence / frame 信息，以及可用的 SMPL 或 SMPL-X 人体先验数据。

### 2. 统一帧号和路径

每个导出的样本都应该能追溯到原始 subject、sequence、camera id、frame id 和文件路径。这里最怕的是能跑，但不知道跑的是哪一帧。

### 3. 检查相机

相机参数是多视角几何的底座。导出前需要明确内参、外参、坐标系约定、图像尺寸和 resize 关系。这里出错，后面的投影、深度、点云都会跟着错。

### 4. 检查 mask

mask 不是可有可无的修饰图。它会影响人体区域、背景保留、投影检查和后续监督。如果 mask 本身有问题，模型结果再好看也需要打问号。

### 5. 对齐人体先验

SMPL / SMPL-X 先验需要通过同一套相机链路投影回图像。projection overlay 在这里只作为诊断图，用来判断先验是否落在合理位置。它不能替代最终的 3D 点云证据。

### 6. 导出 VGGT-ready package

通过检查的 case 可以进入后续路线：一边跑 vanilla VGGT 作为 baseline，一边交给 VGGT-SMPL-X 人体先验分支做模型侧实验。

---

## 导出内容

一个合格的 case package 通常应该包含：

- 多视角 RGB 帧；
- 相机内参和外参；
- human / foreground mask；
- 可用的 SMPL 或 SMPL-X 人体先验信息；
- 用于检查的 projection overlay；
- 必要的 depth / point reference；
- source manifest；
- audit report 或 failure report。

文件结构可以继续演进，但原则不能变：每个产物都要知道来源、用途和边界。

---

## 证据边界

这个项目里必须严格区分几类东西。

| 类型 | 用途 | 能否当最终 student 证据 |
| --- | --- | --- |
| Teacher / reference | 检查几何、辅助监督、帮助定位问题 | 不能 |
| Vanilla VGGT baseline | 后续对比基线 | 不能 |
| Projection overlay / diagnostic | 检查相机、mask、投影是否对齐 | 不能 |
| Student candidate | 下游模型真正吃掉 audited case 后的输出 | 只有通过视觉和对照门控后才可以 |

尤其要避免一个常见误区：二维投影图里人体位置对了，并不代表三维点云就已经有人形；reference 点云更像人，也不代表 student 模型已经学会了人体拓扑。

最终要看的仍然是 human-main full-scene RGB point cloud：人体是画面主体，同时保留一部分环境点，能够和 VGGT baseline / controls 在同一坐标和同一视角下比较。

---

## failure-closed 规则

如果数据没有对齐，就停止导出。

这个时候最有价值的结果不是一张好看的图，而是一份清楚的失败记录，例如：

```text
camera convention mismatch
mask and projected body prior do not overlap
frame index cannot be traced
resized image uses stale intrinsics
teacher/reference artifact is being confused with student output
```

这类失败记录会让后面的实验少走弯路。数据入口已经坏了，就不应该继续训练，再用训练结果解释问题。

---

## 这个仓库的价值

最终图当然重要，但真正让实验可信的，往往是前面这些不显眼的步骤：数据绑定、相机审计、mask 检查、先验投影、manifest、failure report、controls。

这个仓库的价值就在这里：它让后续 VGGT + 人体先验实验能回答一个更关键的问题：

> 如果点云效果变好了，这个提升到底来自人体先验路线，还是来自更干净的数据入口、更合理的 mask，或者某个 reference 产物？

只有数据层先说清楚，后面的模型结果才有讨论价值。

---

## 当前状态

这个仓库目前定位为 research utility / dataset bridge，不作为最终点云重建 benchmark 来宣传。

当前重点是让 ZJU-MoCap 风格 case 能被审计、复用，并能进入更完整的 VGGT + SMPL-X 人体先验实验栈。

---

## 数据说明

这个仓库面向本地 ZJU-MoCap 风格数据和人体模型资产。受限数据集、RGB、mask、camera 文件、SMPL/SMPL-X body model 文件等，不应直接放入公开仓库，除非其许可证明确允许重新分发。

---

## 架构图

架构图文件位置：

```text
docs/figures/vggt_zju_mocap_adapter_architecture.svg
```
