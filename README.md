# VLA Vision Data Analysis

> 面向大规模具身智能（Embodied AI）训练的视觉数据压缩策略分析与最佳实践

## 📋 项目简介

本项目针对 Vision-Language-Action (VLA) 和机器人端到端策略训练场景，系统性地分析了不同图像压缩策略对数据规模、训练性能和信息保真度的影响。

**核心问题**：在规模化采集场景下，如何平衡存储成本、训练 I/O 效率和模型性能？

**核心结论**：JPEG(Q95) 压缩在保持训练有效性的同时，可将存储成本降低一个数量级，且对训练性能影响可忽略。

## 🚀 快速建议（TL;DR）

- **RGB 录制**：`foxglove.CompressedImage`，`jpeg`，`quality=95`（配合 MCAP chunk `zstd`）
- **Depth 录制**：`foxglove.RawImage`，`Z16`（配合 MCAP chunk `zstd`）
- **核心判断**：在 VLA / 机器人端到端策略训练里，JPEG(Q95) 的信息损失通常不会成为性能瓶颈；训练更依赖数据规模与多样性。

> 💡 **注意**：本文提到的 foxglove 与 mcap 均是基于自定义中间件而言，但读者可以自行映射为 ROS2 与 ros2bag。

## 📚 文档结构

### 核心文档

- **[VLA Vision Data Analysis.md](https://github.com/Zwhy2025/VLA-Vision-Data-Analysis/blob/master/VLA%20Vision%20Data%20Analysis.md)** - 完整的技术分析与策略建议
  - 业务与技术背景
  - 压缩策略对比（Raw vs JPEG vs H.264/H.265）
  - 训练友好的数据结构（Zarr）
  - 视频学习与视频压缩的权衡

### 分析报告

- **[JPEG保真度分析.md](https://github.com/Zwhy2025/VLA-Vision-Data-Analysis/blob/master/JPEG保真度分析.md)** - JPEG 压缩质量对图像保真度的定量分析
  - 评估指标：MSE、PSNR、SSIM、MaxAbs
  - 不同质量因子（Q70-Q95）的对比
  - **结论**：JPEG Q95 的 SSIM > 0.99，视觉质量接近无损

- **[图像压缩性能分析.md](https://github.com/Zwhy2025/VLA-Vision-Data-Analysis/blob/master/图像压缩性能分析.md)** - 实机录制时的系统性能基准测试
  - CPU 占用率对比
  - 录制频率（掉帧率）分析
  - 存储压缩比统计
  - **结论**：JPEG 模式在保持 30Hz 满帧率的同时，存储压缩比约 12.8x

## 🛠️ 工具使用

### 图像压缩验证脚本

使用 [`tools/validation_script.py`](https://github.com/Zwhy2025/VLA-Vision-Data-Analysis/blob/master/tools/validation_script.py) 可以评估单张图片在不同压缩策略下的质量损失。

#### 安装依赖

```bash
pip install numpy opencv-python
```

#### 使用方法

```bash
# 在仓库根目录下运行
python3 tools/validation_script.py test_image.png
```

#### 输出内容

脚本会在 `tools/output/` 目录下生成：

1. **压缩后的图像**：`encoded_jpeg_q*.jpg`、`encoded_png_lossless.png`
2. **重建图像**：`recon_jpeg_q*.png`（解码后的图像）
3. **差分热力图**：`diff_jpeg_q*_x8.png`（放大 8 倍以可视化误差）
4. **评估报告**：`tools/report.md`（包含详细的数值指标）

#### 评估指标说明

- **MSE (Mean Squared Error)**：均方误差，越小越好
- **PSNR (Peak Signal-to-Noise Ratio)**：峰值信噪比，>30dB 为优，>40dB 为极优
- **SSIM (Structural Similarity Index)**：结构相似性，范围 0-1，>0.99 意味着人眼几乎无法区分
- **MaxAbs**：最大像素绝对误差

## 📁 项目结构

```
vla_vision_data/
├── README.md                          # 本文件
├── VLA Vision Data Analysis.md        # 核心分析文档
├── JPEG保真度分析.md                  # JPEG 质量分析报告
├── 图像压缩性能分析.md                # 系统性能基准测试报告
├── test_image.png                     # 测试用图像
├── assets/
│   └── images/                        # 性能测试截图
│       ├── p1.png ~ p6.png
└── tools/
    ├── validation_script.py           # 图像压缩验证脚本
    ├── report.md                      # 自动生成的评估报告
    └── output/                        # 脚本输出目录
        ├── encoded_*.jpg/png          # 压缩后的图像
        ├── recon_*.png                # 重建图像
        └── diff_*.png                 # 差分热力图
```

## 🔑 关键发现

### 1. JPEG Q95 是最佳平衡点

- **存储收益**：相比 Raw+Zstd，压缩比约 12.8x
- **质量保证**：SSIM = 0.991，PSNR = 44.76 dB
- **性能影响**：CPU 占用率与 Raw 模式持平（~33%），录制频率无下降

### 2. PNG 不适合实时录制

- CPU 单核跑满（100%）
- 录制频率暴跌至 ~3Hz
- 仅适合低频或离线场景

### 3. 训练友好性优先

- 训练通常需要随机采样帧，而非顺序播放视频
- JPEG 帧内压缩支持快速随机访问
- H.264/H.265 帧间压缩会增加随机访问成本

## 📊 性能对比总结

| 模式 | 压缩比 | SSIM | CPU 占用 | 录制频率 | 随机访问 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Raw + Zstd** | 1.5x | 1.000 | ~33% | 31.6 Hz | ✅ 快 |
| **JPEG Q95 + Zstd** | **12.8x** | **0.991** | **~33%** | **31.6 Hz** | ✅ **快** |
| PNG + Zstd | 15.0x | 1.000 | 100% | ~3 Hz | ✅ 快 |
| H.264/H.265 | 100x+ | ~0.95 | 低 | - | ❌ 慢 |

## 🎯 适用场景

- ✅ **大规模机器人数据采集**：需要长时间录制，存储成本敏感
- ✅ **VLA / 端到端策略训练**：数据规模优先，像素级完美度非瓶颈
- ✅ **随机采样训练**：需要快速随机访问单帧数据
- ❌ **医学影像 / 超分重建**：需要像素级精确的场景
- ❌ **视频回放 / 展示**：需要长视频时，可离线转码为 H.264/H.265

## 📝 延伸阅读

- [训练友好的数据结构：Zarr](https://github.com/Zwhy2025/VLA-Vision-Data-Analysis/blob/master/VLA%20Vision%20Data%20Analysis.md#5-训练传输友好的数据结构zarr推荐作为派生格式)
- [视频学习与视频压缩的权衡](https://github.com/Zwhy2025/VLA-Vision-Data-Analysis/blob/master/VLA%20Vision%20Data%20Analysis.md#6-延伸视频学习与视频压缩)

## 📄 License

本项目采用 MIT License（或根据实际情况调整）。

---

**最后更新**：2026-01-11
