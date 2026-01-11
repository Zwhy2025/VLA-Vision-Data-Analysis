# JPEG 保真度分析

> Quality Validation: Image Fidelity Analysis
>
> 本文档提供关于 JPEG 压缩对图像像素级保真度影响的定量分析。

## 1. 验证目标
量化不同 JPEG 压缩质量（Quality Factor）对图像信息的损失程度，以确定既能大幅减小体积，又能满足 VLA 模型训练需求的最佳压缩参数。

## 2. 评估指标
我们使用以下指标对比 Raw（无损）与 JPEG（有损）图像：
- **MSE (Mean Squared Error)**: 均方误差，越小越好。
- **PSNR (Peak Signal-to-Noise Ratio)**: 峰值信噪比，>30dB 为优，>40dB 为极优。
- **SSIM (Structural Similarity Index)**: 结构相似性，范围 0-1。>0.99 意味着人眼几乎无法区分。
- **MaxAbs**: 最大像素绝对误差。

## 3. 实验结果 (基于典型 Robot Arm 图像)

**输入图片**: `right_arm.png` (640x480, Raw RGB)

| 模式 | 文件大小 (KiB) | 压缩比 | MSE | PSNR (dB) | SSIM | 结论 |
|---|---:|---:|---:|---:|---:|---|
| **Raw (PNG)** | 332.48 | 1.0x | 0.00 | inf | 1.000 | 基准 |
| **JPEG Q95** | **67.28** | **~5.0x** | **2.17** | **44.76** | **0.991** | **推荐** |
| JPEG Q90 | 44.14 | ~7.5x | 3.16 | 43.13 | 0.987 | 可接受 |
| JPEG Q80 | 27.96 | ~11.9x | 4.80 | 41.32 | 0.981 | 轻微失真 |
| JPEG Q70 | 21.96 | ~15.1x | 6.36 | 40.10 | 0.976 | 可见失真 |

### 3.1 详细分析
- **JPEG Q95 (推荐)**:
  - **SSIM = 0.991**：结构信息保留极其完整，远高于一般深度学习数据增强（如 GaussianBlur）引入的失真。
  - **PSNR = 44.76 dB**：信噪比极高，误差仅集中在极高频边缘，且幅度很小（MaxAbs=14，即 0-255 范围内最大偏差仅 14）。
  - **收益**：体积仅为 PNG 的 1/5，Raw Bitmap 的 1/13。

- **JPEG Q90 vs Q95**:
  - Q90 虽然体积更小，但 SSIM 开始下降。考虑到存储成本在 Q95 下已大幅缓解（67KB/帧），为了给模型训练留足余量，我们选择更保守的 **Q95**。

## 4. 验证工具
本报告数据使用 [`tools/validation_script.py`](https://github.com/Zwhy2025/VLA-Vision-Data-Analysis/blob/master/tools/validation_script.py) 生成。

### 运行方式
```bash
# 在仓库根目录（vla_vision_data）下
python3 tools/validation_script.py test_image.png
```

该脚本会生成：
1. 不同质量的 JPEG 压缩图。
2. 解码后的重建图 (Reconstructed)。
3. 差分热力图 (Difference Map, 放大 8 倍以可视化误差)。
