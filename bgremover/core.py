"""rembg wrapper with preset management and session caching."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from PIL import Image
from rembg import new_session, remove

from .postprocess import autocrop, composite_background, despeckle, feather_edges

SUPPORTED_MODELS = [
    "birefnet-general",
    "isnet-general-use",
    "u2net",
    "sam",
]

PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "model": "u2net",
        "alpha_matting": False,
        "alpha_matting_fg_threshold": 240,
        "alpha_matting_bg_threshold": 10,
        "alpha_matting_erode_size": 10,
        "feather": False,
        "feather_radius": 1.5,
        "despeckle_size": 0,
        "do_autocrop": True,
        "crop_padding": 40,
        "background": None,
    },
    "balanced": {
        "model": "isnet-general-use",
        "alpha_matting": True,
        "alpha_matting_fg_threshold": 240,
        "alpha_matting_bg_threshold": 10,
        "alpha_matting_erode_size": 10,
        "feather": True,
        "feather_radius": 1.5,
        "despeckle_size": 50,
        "do_autocrop": True,
        "crop_padding": 40,
        "background": None,
    },
    "best": {
        "model": "birefnet-general",
        "alpha_matting": True,
        "alpha_matting_fg_threshold": 240,
        "alpha_matting_bg_threshold": 10,
        "alpha_matting_erode_size": 10,
        "feather": True,
        "feather_radius": 1.5,
        "despeckle_size": 50,
        "do_autocrop": True,
        "crop_padding": 40,
        "background": None,
    },
}

# Module-level session cache — avoids re-downloading weights on repeated calls.
_session_cache: Dict[str, Any] = {}


@dataclass
class ProcessConfig:
    model: str = "birefnet-general"
    alpha_matting: bool = True
    alpha_matting_fg_threshold: int = 240
    alpha_matting_bg_threshold: int = 10
    alpha_matting_erode_size: int = 10
    feather: bool = True
    feather_radius: float = 1.5
    despeckle_size: int = 50
    do_autocrop: bool = True
    crop_padding: int = 40
    background: Optional[str] = None  # None=transparent, "white", "black", "#RRGGBB"

    @classmethod
    def from_preset(cls, name: str) -> "ProcessConfig":
        if name not in PRESETS:
            raise ValueError(f"Unknown preset '{name}'. Choose from: {list(PRESETS)}")
        return cls(**PRESETS[name])

    def apply_overrides(self, **overrides: Any) -> "ProcessConfig":
        """Return a new config with only the non-None overrides applied."""
        import copy
        cfg = copy.copy(self)
        for k, v in overrides.items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


def get_session(model_name: str) -> Any:
    if model_name not in _session_cache:
        _session_cache[model_name] = new_session(model_name)
    return _session_cache[model_name]


def process_image(image: Image.Image, config: ProcessConfig) -> Image.Image:
    """Run the full background-removal pipeline on a PIL image."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    session = get_session(config.model)

    # SAM already produces clean masks; alpha matting adds noise on top of it.
    use_matting = config.alpha_matting and config.model != "sam"

    try:
        result: Image.Image = remove(
            image,
            session=session,
            alpha_matting=use_matting,
            alpha_matting_foreground_threshold=config.alpha_matting_fg_threshold,
            alpha_matting_background_threshold=config.alpha_matting_bg_threshold,
            alpha_matting_erode_size=config.alpha_matting_erode_size,
        )
    except MemoryError:
        if use_matting:
            import warnings
            warnings.warn(
                "Alpha matting ran out of memory (image too large); retrying without it.",
                RuntimeWarning,
                stacklevel=2,
            )
            result = remove(
                image,
                session=session,
                alpha_matting=False,
            )
        else:
            raise

    if config.despeckle_size > 0:
        result = despeckle(result, min_size=config.despeckle_size)

    if config.feather:
        result = feather_edges(result, radius=config.feather_radius)

    if config.do_autocrop:
        result = autocrop(result, padding=config.crop_padding)

    if config.background:
        result = composite_background(result, color=config.background)

    return result
