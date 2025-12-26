#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Dilate → Waifu2x Upscale → Erode Pipeline (GPU Vulkan)
================================================================================

This script implements a complete image processing pipeline consisting of:

1. Morphological Dilation (OpenCV)
2. Image Upscaling & Denoising using Waifu2x (NCNN + Vulkan GPU backend)
3. Morphological Erosion (OpenCV)

The goal of this pipeline is to:
- Slightly expand fine structures using dilation
- Enhance resolution and suppress noise using waifu2x
- Refine boundaries using erosion after upscaling

All morphological parameters (kernel size and iteration count) are kept
exactly the same as provided by the user.

Key Features:
-------------
- Fully GPU-accelerated waifu2x inference via Vulkan
- Precise inference-time measurement (milliseconds)
- Deterministic and reproducible image processing
- Clean and modular structure suitable for GitHub publication
- Ideal for research, industrial vision, and preprocessing pipelines

Processing Order:
-----------------
Input Image
   → Dilate (1x1 kernel, 1 iteration)
   → Waifu2x Upscale (2x, noise=3)
   → Erode (2x2 kernel, 1 iteration)
   → Final Output Image

Requirements:
-------------
- Python 3.9 or 3.10
- waifu2x-ncnn-vulkan-python
- Pillow (PIL)
- OpenCV (cv2)
- NumPy

Author:
-------
Furkan KARAKAYA
GitHub: F-Karakaya
"""

import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from waifu2x_ncnn import Waifu2x


def dilate_image(input_path: Path, output_path: Path) -> None:
    """
    Apply morphological dilation to a grayscale image.

    Parameters are intentionally fixed:
    - Kernel size: (1, 1)
    - Iterations: 1
    """

    img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    kernel = np.ones((1, 1), dtype=np.uint8)
    result = cv2.dilate(img, kernel, iterations=1)

    cv2.imwrite(str(output_path), result)


def erode_image(input_path: Path, output_path: Path) -> None:
    """
    Apply morphological erosion to a grayscale image.

    Parameters are intentionally fixed:
    - Kernel size: (2, 2)
    - Iterations: 1
    """

    img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    kernel = np.ones((2, 2), dtype=np.uint8)
    result = cv2.erode(img, kernel, iterations=1)

    cv2.imwrite(str(output_path), result)


def main(input_path: Path, output_path: Path) -> None:
    """
    Main pipeline execution function.

    Steps:
    1. Apply dilation
    2. Upscale the dilated image using Waifu2x (GPU)
    3. Apply erosion to the upscaled result
    4. Report inference time
    """

    work_dir = input_path.parent

    dilated_path = work_dir / f"{input_path.stem}-dilate1.png"
    waifu2x_path = work_dir / f"{input_path.stem}-dilate1-waifu2x.png"
    eroded_path = output_path

    # ---------------------------------------------------------------------
    # Step 1: Dilate
    # ---------------------------------------------------------------------
    print("🔧 Step 1: Applying dilation...")
    dilate_image(input_path, dilated_path)

    # ---------------------------------------------------------------------
    # Step 2: Waifu2x Upscale (GPU)
    # ---------------------------------------------------------------------
    print("🚀 Step 2: Running waifu2x (GPU Vulkan)...")

    waifu2x = Waifu2x(
        gpuid=0,
        scale=2,
        noise=3,
        tilesize=200,
        num_threads=2
    )

    if waifu2x._gpuid == -1:
        print("⚠️  Waifu2x running in CPU mode")
    else:
        print(f"✅ Waifu2x running in GPU mode (GPU ID = {waifu2x._gpuid})")

    image = Image.open(dilated_path).convert("RGB")

    start_time = time.perf_counter()
    output_image = waifu2x.process(image)
    end_time = time.perf_counter()

    elapsed_ms = (end_time - start_time) * 1000.0

    output_image.save(waifu2x_path)

    print(f"🕒 Waifu2x inference time: {elapsed_ms:.2f} ms")

    # ---------------------------------------------------------------------
    # Step 3: Erode
    # ---------------------------------------------------------------------
    print("🔧 Step 3: Applying erosion...")
    erode_image(waifu2x_path, eroded_path)

    # ---------------------------------------------------------------------
    # Done
    # ---------------------------------------------------------------------
    print("🎉 Pipeline completed successfully")
    print(f"📁 Final output saved to: {eroded_path}")


if __name__ == "__main__":

    main(
        input_path=Path("data/input-image.png"),
        output_path=Path("data/input-image-dilate1-waifu2x-erode2.png")
    )
