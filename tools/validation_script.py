#!/usr/bin/env python3
"""
对单张图片做“压缩 -> 还原”评估，并输出报告。

目的：
- 量化 JPEG（有损）在不同 quality 下对原图的影响
- 对比 PNG（无损）作为参考

输出：
- 控制台：每种模式的 size / PSNR / SSIM / MSE / MaxAbs
- 文件：重建图、差分图（便于直观看差异）

依赖：
- numpy
- opencv-python (cv2)

用法：
    python3 validation_script.py <input_image>
    
示例：
    python3 validation_script.py ../test_image.png
"""

from __future__ import annotations

import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

# ============================================================================
# 可调整参数（全局变量）
# ============================================================================

# JPEG质量参数（默认95）
JPEG_QUALITY = 95

# 差分图放大倍数（便于肉眼观察细微差异）
DIFF_AMPLIFY = 8.0

# 输出目录（相对于脚本所在目录，脚本已在 compression_images/ 目录下）
OUTPUT_DIR = "output"

# 额外评估的JPEG质量值（除了默认的JPEG_QUALITY）
EXTRA_JPEG_QUALITIES = [90, 80, 70]


def _read_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def _imencode(ext: str, img_bgr: np.ndarray, params: list[int]) -> bytes:
    ok, buf = cv2.imencode(ext, img_bgr, params)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for {ext}")
    return buf.tobytes()


def _imdecode(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imdecode failed")
    return img


def mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff * diff))


def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    m = mse(a, b)
    if m == 0.0:
        return float("inf")
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(m)


def _gaussian_kernel_2d(ksize: int = 11, sigma: float = 1.5) -> np.ndarray:
    # OpenCV 的 getGaussianKernel 返回列向量
    g1 = cv2.getGaussianKernel(ksize, sigma)
    g2 = g1 @ g1.T
    return g2.astype(np.float64)


def ssim_gray(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    """
    计算单通道 SSIM（与 skimage 的结构类似，便于量化对比）。
    """
    if a_gray.shape != b_gray.shape:
        raise ValueError("SSIM inputs must have same shape")
    if a_gray.ndim != 2:
        raise ValueError("SSIM inputs must be grayscale 2D arrays")

    a = a_gray.astype(np.float64)
    b = b_gray.astype(np.float64)

    kernel = _gaussian_kernel_2d(ksize=11, sigma=1.5)

    mu_a = cv2.filter2D(a, -1, kernel, borderType=cv2.BORDER_REFLECT)
    mu_b = cv2.filter2D(b, -1, kernel, borderType=cv2.BORDER_REFLECT)

    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a2 = cv2.filter2D(a * a, -1, kernel, borderType=cv2.BORDER_REFLECT) - mu_a2
    sigma_b2 = cv2.filter2D(b * b, -1, kernel, borderType=cv2.BORDER_REFLECT) - mu_b2
    sigma_ab = cv2.filter2D(a * b, -1, kernel, borderType=cv2.BORDER_REFLECT) - mu_ab

    # 常用默认：L=255, k1=0.01, k2=0.03
    L = 255.0
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2

    num = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    ssim_map = num / (den + 1e-12)
    return float(np.mean(ssim_map))


def ssim_color(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    """
    简单做法：转灰度后计算 SSIM。
    这不等价于“彩色多通道 SSIM”，但足以作为工程上的快速指标。
    """
    a_gray = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)
    return ssim_gray(a_gray, b_gray)


def max_abs(a: np.ndarray, b: np.ndarray) -> int:
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return int(np.max(diff))


def diff_vis(a_bgr: np.ndarray, b_bgr: np.ndarray, amplify: float = 8.0) -> np.ndarray:
    """
    差分可视化：absdiff 后放大（便于肉眼观察细微差异）。
    """
    d = cv2.absdiff(a_bgr, b_bgr).astype(np.float32) * float(amplify)
    d = np.clip(d, 0, 255).astype(np.uint8)
    return d


@dataclass
class ReportItem:
    name: str
    size_bytes: int
    mse: float
    psnr: float
    ssim: float
    max_abs: int


def evaluate_variant(
    name: str,
    encoded_bytes: bytes,
    original_bgr: np.ndarray,
) -> Tuple[ReportItem, np.ndarray]:
    recon_bgr = _imdecode(encoded_bytes)
    item = ReportItem(
        name=name,
        size_bytes=len(encoded_bytes),
        mse=mse(original_bgr, recon_bgr),
        psnr=psnr(original_bgr, recon_bgr),
        ssim=ssim_color(original_bgr, recon_bgr),
        max_abs=max_abs(original_bgr, recon_bgr),
    )
    return item, recon_bgr


def generate_report(
    report_path: Path,
    in_path: Path,
    orig_size_bytes: int,
    orig: np.ndarray,
    orig_format: str,
    orig_channels: int,
    results: list[ReportItem],
) -> None:
    """
    生成压缩评估报告到文件。

    Args:
        report_path: 报告文件路径
        in_path: 输入图片路径
        orig_size_bytes: 原始图片大小（字节）
        orig: 原始图片（BGR格式，用于获取尺寸）
        orig_format: 原始图片格式（RGBA/RGB/GRAY）
        orig_channels: 原始图片通道数
        results: 评估结果列表
    """
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 图像压缩评估报告\n")


        # 输入图片信息
        f.write("## 输入图片信息\n\n")
        f.write(f"- 路径: {in_path}\n")
        f.write(f"- 原始大小: {orig_size_bytes / 1024:.2f} KiB ({orig_size_bytes} bytes)\n")
        f.write(f"- 尺寸: {orig.shape[1]} x {orig.shape[0]} 像素\n")
        f.write(f"- 原始格式: {orig_format} ({orig_channels} 通道)\n")
        f.write(f"- 评估用格式: RGB (3通道，BGR顺序，用于与JPEG/PNG对比)\n")
        f.write(f"- 注意: 原始文件已直接复制到 output/{in_path.name}，保留原始格式和大小\n\n")

        # 指标表格
        f.write("## 压缩模式对比\n\n")
        f.write("| 模式 | 文件大小 (KiB) | MSE | PSNR (dB) | SSIM | MaxAbs |\n")
        f.write("|------|---------------|-----|-----------|------|--------|\n")
        for r in results:
            psnr_str = f"{r.psnr:.3f}" if r.psnr != float("inf") else "inf"
            f.write(
                f"| {r.name} | {r.size_bytes/1024:.2f} | {r.mse:.4f} | {psnr_str} | {r.ssim:.6f} | {r.max_abs} |\n"
            )
        f.write("\n")

        # 文件大小对比
        f.write("## 文件大小对比\n\n")
        png_lossless = next((r for r in results if r.name == "png_lossless"), None)
        if png_lossless:
            f.write(f"- PNG (无损): {png_lossless.size_bytes / 1024:.2f} KiB (基准)\n")
            for r in results:
                if r.name != "png_lossless":
                    ratio = png_lossless.size_bytes / r.size_bytes if r.size_bytes > 0 else 0
                    f.write(f"- {r.name}: {r.size_bytes / 1024:.2f} KiB (相对PNG: {ratio:.2f}x)\n")
        f.write("\n")

        # 压缩比统计（相对于原始文件）
        f.write("## 压缩比统计（相对于原始文件）\n\n")
        for r in results:
            ratio = orig_size_bytes / r.size_bytes if r.size_bytes > 0 else 0
            reduction_pct = (1 - r.size_bytes / orig_size_bytes) * 100 if orig_size_bytes > 0 else 0
            f.write(f"- {r.name}: {ratio:.2f}x 压缩 ({reduction_pct:.1f}% 减少)\n")
        f.write("\n")

        # 质量指标说明
        f.write("## 质量指标说明\n\n")
        f.write("- **MSE (Mean Squared Error)**: 均方误差，越小越好，0表示完全一致\n")
        f.write("- **PSNR (Peak Signal-to-Noise Ratio)**: 峰值信噪比，越大越好，inf表示完全一致\n")
        f.write("- **SSIM (Structural Similarity Index)**: 结构相似性，范围0-1，1表示完全一致\n")
        f.write("- **MaxAbs**: 最大绝对误差（像素值），越小越好，0表示完全一致\n")
        f.write("\n")

        # 结论和建议
        f.write("## 结论和建议\n\n")
        jpeg_q95 = next((r for r in results if r.name.startswith("jpeg_q95")), None)
        if jpeg_q95:
            f.write(f"- **JPEG质量95**（默认）:\n")
            f.write(f"  - 文件大小: {jpeg_q95.size_bytes / 1024:.2f} KiB\n")
            f.write(f"  - PSNR: {jpeg_q95.psnr:.3f} dB\n")
            f.write(f"  - SSIM: {jpeg_q95.ssim:.6f}\n")
            f.write(f"  - 压缩比: {orig_size_bytes / jpeg_q95.size_bytes:.2f}x\n")
            if jpeg_q95.ssim > 0.99:
                f.write(f"  - 评估: SSIM > 0.99，视觉质量接近无损\n")
            elif jpeg_q95.ssim > 0.95:
                f.write(f"  - 评估: SSIM > 0.95，视觉质量良好，适合大多数应用\n")
            else:
                f.write(f"  - 评估: SSIM < 0.95，可能存在明显视觉差异\n")
            f.write("\n")

        f.write("- **PNG (无损)**:\n")
        if png_lossless:
            f.write(f"  - 文件大小: {png_lossless.size_bytes / 1024:.2f} KiB\n")
            f.write(f"  - 完全无损，适合需要精确像素值的场景\n")
        f.write("\n")

        f.write("- **建议**:\n")
        f.write("  - 对于机器人数据录制，JPEG质量95在存储空间和视觉质量之间取得良好平衡\n")
        f.write("  - 如需完全无损，使用PNG（但文件更大）\n")
        f.write("  - 查看 output/ 目录中的 diff_*.png 文件可直观看到压缩损失的位置\n")
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write(f"报告生成时间: {os.popen('date').read().strip()}\n")
        f.write("=" * 70 + "\n")


def main() -> int:
    # 解析命令行参数：只接受一个输入图片路径
    if len(sys.argv) != 2:
        print("用法: python3 validation_script.py <input_image>")
        return 1
    else:
        in_path = Path(sys.argv[1])
        # 如果是相对路径，相对于脚本所在目录解析
        if not in_path.is_absolute():
            in_path = Path(__file__).parent / in_path

    if not in_path.exists():
        print(f"错误: 文件不存在: {in_path}")
        return 1

    # 使用全局变量配置
    # 脚本在 compression_images/ 目录下，输出目录为 compression_images/output/
    base_out_dir = Path(__file__).parent
    output_dir = base_out_dir / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取原始图片信息（在读取前）
    orig_size_bytes = in_path.stat().st_size
    orig_img_info = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
    if orig_img_info is None:
        raise FileNotFoundError(f"Failed to read image: {in_path}")
    orig_channels = orig_img_info.shape[2] if orig_img_info.ndim == 3 else 1
    orig_format = "RGBA" if orig_channels == 4 else ("RGB" if orig_channels == 3 else "GRAY")

    # 读取为BGR用于评估（丢弃alpha通道，但原始文件会直接复制）
    orig = _read_image_bgr(in_path)

    variants: Dict[str, bytes] = {}

    # PNG（无损）作为基线
    variants["png_lossless"] = _imencode(".png", orig, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    # JPEG（有损）- 使用全局变量配置
    q = max(1, min(100, JPEG_QUALITY))
    variants[f"jpeg_q{q}"] = _imencode(".jpg", orig, [cv2.IMWRITE_JPEG_QUALITY, q])

    # 额外评估的JPEG质量值（便于对比）
    for q2 in EXTRA_JPEG_QUALITIES:
        if q2 == q:
            continue
        variants[f"jpeg_q{q2}"] = _imencode(".jpg", orig, [cv2.IMWRITE_JPEG_QUALITY, q2])

    # 评估并落盘
    results: list[ReportItem] = []

    # 直接复制原图到输出目录（保留原始格式和大小）
    original_copy_path = output_dir / in_path.name
    shutil.copy2(in_path, original_copy_path)

    for name, data in variants.items():
        item, recon = evaluate_variant(name, data, orig)
        results.append(item)

        # 重建图
        recon_path = output_dir / f"recon_{name}.png"
        cv2.imwrite(str(recon_path), recon)

        # 差分图
        dimg = diff_vis(orig, recon, amplify=DIFF_AMPLIFY)
        diff_path = output_dir / f"diff_{name}_x{DIFF_AMPLIFY:g}.png"
        cv2.imwrite(str(diff_path), dimg)

        # 编码后的文件也存一份（用于查看真实体积）
        ext = ".png" if name.startswith("png") else ".jpg"
        (output_dir / f"encoded_{name}{ext}").write_bytes(data)

    # 打印报告到控制台
    results.sort(key=lambda x: x.size_bytes)
    print(f"Input: {in_path}  ({orig.shape[1]}x{orig.shape[0]})")
    print(f"Output dir: {output_dir}")
    print()
    print("name,size_kib,mse,psnr_db,ssim_gray,max_abs")
    for r in results:
        print(
            f"{r.name},{r.size_bytes/1024:.2f},{r.mse:.4f},{r.psnr:.3f},{r.ssim:.6f},{r.max_abs}"
        )

    # 生成 report.md
    report_path = base_out_dir / "report.md"
    generate_report(
        report_path=report_path,
        in_path=in_path,
        orig_size_bytes=orig_size_bytes,
        orig=orig,
        orig_format=orig_format,
        orig_channels=orig_channels,
        results=results,
    )
    print(f"\n报告已保存到: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
