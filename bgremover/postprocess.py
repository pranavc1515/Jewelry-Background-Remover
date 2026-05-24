"""Post-processing steps applied after rembg mask extraction."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter


def despeckle(image: Image.Image, min_size: int = 50) -> Image.Image:
    """Drop alpha-masked connected components smaller than min_size pixels.

    Removes dust, stray reflections, or tag fragments that survive segmentation.
    """
    if min_size <= 0:
        return image

    arr = np.array(image.convert("RGBA"))
    alpha = arr[:, :, 3]
    binary = (alpha > 128).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    keep_mask = np.zeros_like(binary)
    for label_id in range(1, num_labels):  # label 0 is background
        if stats[label_id, cv2.CC_STAT_AREA] >= min_size:
            keep_mask[labels == label_id] = 1

    arr[:, :, 3] = arr[:, :, 3] * keep_mask
    return Image.fromarray(arr, mode="RGBA")


def feather_edges(image: Image.Image, radius: float = 1.5) -> Image.Image:
    """Soften alpha edges with a Gaussian blur — only the alpha channel is touched."""
    r, g, b, a = image.convert("RGBA").split()
    a = a.filter(ImageFilter.GaussianBlur(radius=radius))
    return Image.merge("RGBA", (r, g, b, a))


def autocrop(image: Image.Image, padding: int = 40) -> Image.Image:
    """Crop the image to the bounding box of non-transparent pixels plus padding."""
    arr = np.array(image.convert("RGBA"))
    alpha = arr[:, :, 3]

    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)

    if not rows.any():
        return image  # fully transparent — return as-is

    rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

    h, w = alpha.shape
    rmin = max(0, rmin - padding)
    rmax = min(h, rmax + padding + 1)
    cmin = max(0, cmin - padding)
    cmax = min(w, cmax + padding + 1)

    return Image.fromarray(arr[rmin:rmax, cmin:cmax], mode="RGBA")


def composite_background(image: Image.Image, color: str = "white") -> Image.Image:
    """Flatten transparent PNG onto a solid background.

    color: "white", "black", or a hex string like "#F0E8D0".
    Returns an RGB image (no alpha).
    """
    image = image.convert("RGBA")

    if color.lower() == "white":
        bg_rgb: Tuple[int, int, int] = (255, 255, 255)
    elif color.lower() == "black":
        bg_rgb = (0, 0, 0)
    else:
        hex_str = color.lstrip("#")
        if len(hex_str) != 6:
            raise ValueError(f"Invalid color '{color}'. Use 'white', 'black', or '#RRGGBB'.")
        bg_rgb = tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[assignment]

    background = Image.new("RGBA", image.size, bg_rgb + (255,))
    background.paste(image, mask=image.split()[3])
    return background.convert("RGB")
