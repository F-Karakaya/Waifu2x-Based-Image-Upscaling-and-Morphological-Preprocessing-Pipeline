#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Waifu2x NCNN Vulkan Python Example (with Timing)
================================================================================

This script demonstrates how to use the `waifu2x-ncnn-vulkan-python` wrapper
to perform image upscaling and denoising using GPU acceleration via Vulkan.

The underlying waifu2x implementation is based on the NCNN framework and
supports Intel, AMD, and NVIDIA GPUs without requiring CUDA, TensorFlow,
or PyTorch.

In addition to basic usage, this script also measures and reports the
processing (inference) time in milliseconds, which is useful for
benchmarking, performance analysis, and academic experiments.

Key Features:
-------------
- Pure Python API (no external executable calls)
- GPU acceleration via Vulkan (automatic CPU fallback if needed)
- Configurable scale, noise level, tile size, and thread count
- Precise inference-time measurement in milliseconds (ms)
- Clean, minimal, and production-ready structure
- Suitable for GitHub publication, research, and industrial pipelines

Typical Use Cases:
------------------
- Image super-resolution and denoising
- Preprocessing for computer vision pipelines
- Dataset enhancement
- Performance benchmarking (CPU vs GPU)

Requirements:
-------------
- Python 3.9 or 3.10 (must match the installed wheel)
- waifu2x-ncnn-vulkan-python
- Pillow (PIL)

Author:
-------
Furkan KARAKAYA / GitHub Username: F-Karakaya
"""

import time
from PIL import Image
from waifu2x_ncnn import Waifu2x
from pathlib import Path

def main(input_path: Path, output_path: Path) -> None:
    """
    Main execution function.

    This function:
    1. Initializes the Waifu2x engine
    2. Reports whether GPU or CPU mode is used
    3. Loads an input image
    4. Runs waifu2x inference while measuring execution time
    5. Saves the output image
    """

    # -------------------------------------------------------------------------
    # Initialize Waifu2x
    # -------------------------------------------------------------------------
    # gpuid:
    #   0  -> use first available GPU
    #  -1  -> force CPU mode
    #
    # scale:
    #   1 = no upscaling
    #   2 = 2x upscaling
    #
    # noise:
    #  -1 = no denoise
    #   0–3 = increasing denoise strength
    #
    # tilesize:
    #   Smaller values reduce GPU memory usage (recommended for low VRAM GPUs)
    #
    # num_threads:
    #   Number of processing threads (usually 1–2 is sufficient)
    # -------------------------------------------------------------------------

    waifu2x = Waifu2x(
        gpuid=0,
        scale=2,
        noise=3,
        tilesize=200,
        num_threads=2
    )

    # -------------------------------------------------------------------------
    # Report execution mode
    # -------------------------------------------------------------------------
    if waifu2x._gpuid == -1:
        print("⚠️  Running in CPU mode")
    else:
        print(f"✅ Running in GPU mode (GPU ID = {waifu2x._gpuid})")

    image = Image.open(input_path).convert("RGB")

    # -------------------------------------------------------------------------
    # Run waifu2x inference and measure time
    # -------------------------------------------------------------------------
    try:
        print(f"🚀 Starting waifu2x processing...")        
        start_time = time.perf_counter()
        output_image = waifu2x.process(image)
        end_time = time.perf_counter()
    except Exception as exc:
        print(f"❌ Error during waifu2x processing: {exc}")
        raise

    # Calculate elapsed time in milliseconds
    elapsed_ms = (end_time - start_time) * 1000.0

    # -------------------------------------------------------------------------
    # Save result
    # -------------------------------------------------------------------------
    output_image.save(output_path)

    # -------------------------------------------------------------------------
    # Report results
    # -------------------------------------------------------------------------
    print(f"🎉 Processing completed successfully")
    print(f"🕒 Inference time: {elapsed_ms:.2f} ms")
    print(f"📁 Output saved to: {output_path}")


if __name__ == "__main__":

    main(
        input_path=Path("data/input-image.png"),
        output_path=Path("data/input-image-waifu2x.png")
    )
