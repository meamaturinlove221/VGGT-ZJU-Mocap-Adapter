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
  <a href="#并行工程补充">并行工程补充</a> ·
  <a href="#当前成果快照">成果快照</a>
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

多视角人体重建里常见情况包括：

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

## 并行工程补充

这份 ZJU-MoCap 适配工作后来被放进更大的并行工程里一起复盘：任务从“把 SMPL-X 接入 VGGT”推进到了“sparse-view 人体高质量几何恢复的工程闭环”。这一阶段已经跑通了不少链路，但也明确暴露了 6-view head / face 点云质量的上限。

主链路可以概括成四层：

1. **Pose-aligned SMPL-X driver**：读取 pose / shape / expression / translation / scale，把参数化人体放到当前姿态和场景坐标里。
2. **Dense prior maps**：把 posed mesh 投到真实相机下，生成逐视角对齐的 dense prior，包括 depth、camera/world points、normal、visibility、canonical coordinates、body-part features 等。
3. **Input-side / layer-wise fusion**：RGB 提供真实外观和背景，prior maps 提供 pose-aligned 几何位置，mask 限制人体先验的作用范围。先验不只在输入端拼一次，而是在多层特征演化过程中持续参与。
4. **Output-side supervision**：训练侧支持 depth / point / normal / point-normal 等几何监督，也支持 ROI 和 boundary 加权。

SMPL / SMPL-X 在这里的角色不是最终结果，而是 pose-aligned geometry prior。它提供人体大体位置、深度、表面方向和区域约束；真正要证明的仍然是下游模型是否能在 sparse-view 条件下生成更清晰、连续、稳定的 3D 人体点云。

这一阶段的经验也说明：只增加 loss 或者让 ROI 点数变多，并不等于几何质量提升。如果 teacher 本身不够连续、对齐，可见面也不完整，就容易出现“点数增加但 Open3D 更差”的伪阳性。

---

## 已排查路线和失败边界

并行实验里排查过多条方向：

- projected targetpatch / summary-token patch；
- 从同一 checkpoint 继续做 point-normal / humancrop 微调；
- TeacherGeom / ROI combo；
- confidence-collapse pseudo-positive，也就是 face ROI 点数看起来暴涨，但 confidence threshold 或 Open3D 评估反而说明质量更差；
- NormalBae、Sapiens、DepthAnything、DepthPro 等外部 teacher 路线。

这些排查得到的结论比较明确：当前瓶颈不是缺少脚本，而是缺少足够高质量、连续、对齐的 head / face geometry teacher，或者缺少一种能直接改善 sparse-view target-view surface 的局部几何优化方法。

所以后续路线必须转向更硬的几件事：

- real 3D learned residual；
- multi-view detail supervision；
- baseline high-confidence detail preservation；
- SMPL feature-conditioned local geometry branch；
- human-main full-scene visual gate。

---

## 当前成果快照

<p align="center">
  <img src="docs/figures/yuque_parallel_face_head_results.svg" alt="6-view face/head ROI result grid" width="72%" />
</p>

<p align="center"><sub>6-view face/head ROI 复核结果：已经能看到局部面部结构，但仍然存在连续性和稳定性问题。</sub></p>

<p align="center">
  <img src="docs/figures/yuque_kinect_fusion_control_grid.svg" alt="Kinect direct fusion control grid" width="100%" />
</p>

<p align="center"><sub>Kinect direct fusion 保守参数对照：作为外部几何路线排查记录，不作为 student 输出。</sub></p>

目前较安全的结论是：6 视角下已经取得了不错的局部面部结果，但仍然有瑕疵；同协议 6-view face / head 点云还没有达到足够清晰、连续、稳定的最终要求。

---

## 这个仓库的价值

基于 ZJU-MoCap 数据集的工作，为我们熟悉 VGGT 和尝试人体前馈先验提供了台阶。它把很多容易被忽略的工程问题提前暴露出来：数据绑定、背景干扰、相机链路、mask 质量、诊断图误判、点云主图评价方式等。

这些问题推动了后续路线调整。真正缺的不是某个单独脚本，而是更可靠的模型表示、训练目标、局部细节生成和 3D 主图评估体系。

在导师建议下，后续工作引入了 4K4D 数据集和 SMPL-X，并转向更完整的 VGGT + SMPL-X 人体先验实验栈。

---

## 当前状态

这个仓库目前定位为 research utility / dataset bridge，不作为最终点云重建 benchmark 来宣传。

当前重点是让 ZJU-MoCap 风格 case 能被审计、复用，并保留并行工程阶段的排查记录、成果快照和失败边界。

---

## 数据说明

这个仓库面向本地 ZJU-MoCap 风格数据和人体模型资产。受限数据集、RGB、mask、camera 文件、SMPL/SMPL-X body model 文件等，不应直接放入公开仓库，除非其许可证明确允许重新分发。

---

## 架构图与配图

架构图与本次补充配图文件位置：

```text
docs/figures/vggt_zju_mocap_adapter_architecture.svg
docs/figures/yuque_parallel_face_head_results.svg
docs/figures/yuque_kinect_fusion_control_grid.svg
```
