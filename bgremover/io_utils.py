"""Image I/O utilities: EXIF, HEIC support, and batch loading."""
from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional

from PIL import Image, ImageOps

# Register HEIC/HEIF support before any Image.open calls.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _heic_available = True
except ImportError:
    _heic_available = False

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tiff", ".tif"}


def load_image(path: Path) -> Image.Image:
    """Open an image, apply EXIF rotation, and return as RGBA."""
    if not _heic_available and Path(path).suffix.lower() in {".heic", ".heif"}:
        raise RuntimeError(
            "HEIC support requires pillow-heif. Install with:\n"
            "  pip install pillow-heif"
        )
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # honour camera orientation metadata
    return img.convert("RGBA")


def save_image(image: Image.Image, path: Path) -> None:
    """Save a PIL image as a lossless PNG, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path), format="PNG")


def iter_images(folder: Path) -> Generator[Path, None, None]:
    """Yield all supported image paths found directly inside folder (non-recursive)."""
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield p


def output_path(
    input_path: Path,
    output_dir: Optional[Path],
    suffix: str = "_nobg",
) -> Path:
    """Compute the PNG output path for a given input file."""
    stem = input_path.stem + suffix
    filename = stem + ".png"
    if output_dir is not None:
        return output_dir / filename
    return input_path.parent / filename
