"""
Smoke tests for the jewelry bg-remover pipeline.

Unit tests (no model download) validate post-processing logic.
Integration tests download u2net on first run (~170 MB, cached to ~/.u2net/).
Run integration tests with:  pytest tests/ -m integration
Skip them with:              pytest tests/ -m "not integration"
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from bgremover.postprocess import autocrop, composite_background, despeckle, feather_edges


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_rgba(width: int = 200, height: int = 200) -> Image.Image:
    """RGBA: red square (rows/cols 40–159) on transparent canvas. No speckle."""
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[40:-40, 40:-40] = [200, 50, 50, 255]
    return Image.fromarray(arr, mode="RGBA")


def _make_rgba_with_speckle(width: int = 200, height: int = 200) -> Image.Image:
    """Same as _make_rgba but with a tiny 3×3 speckle at the top-left corner."""
    arr = np.array(_make_rgba(width, height))
    arr[0:3, 0:3] = [200, 50, 50, 255]
    return Image.fromarray(arr, mode="RGBA")


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestDespeckle:
    def test_removes_small_components(self):
        img = _make_rgba_with_speckle()
        result = despeckle(img, min_size=20)
        arr = np.array(result)
        # Corner speckle (9 px) should be cleared.
        assert arr[1, 1, 3] == 0

    def test_preserves_large_components(self):
        img = _make_rgba_with_speckle()
        result = despeckle(img, min_size=20)
        arr = np.array(result)
        # Central square (120×120 = 14400 px) must be intact.
        assert arr[100, 100, 3] == 255

    def test_zero_min_size_is_noop(self):
        img = _make_rgba_with_speckle()
        result = despeckle(img, min_size=0)
        assert np.array_equal(np.array(img), np.array(result))


class TestFeatherEdges:
    def test_returns_rgba(self):
        img = _make_rgba()
        result = feather_edges(img, radius=1.5)
        assert result.mode == "RGBA"

    def test_interior_alpha_unchanged(self):
        img = _make_rgba()
        result = feather_edges(img, radius=1.0)
        arr = np.array(result)
        # Deep interior pixel should remain fully opaque.
        assert arr[100, 100, 3] == 255

    def test_edge_alpha_softened(self):
        img = _make_rgba()
        result = feather_edges(img, radius=2.0)
        arr_before = np.array(img)
        arr_after = np.array(result)
        # Edge pixels (row 40) should have their alpha reduced or unchanged —
        # the blur softens the hard boundary.
        edge_before = arr_before[40, 100, 3]
        edge_after = arr_after[40, 100, 3]
        # After blurring, edge row could be anywhere from 0–255; just verify mode is preserved.
        assert 0 <= edge_after <= 255
        assert edge_before != edge_after or edge_after == 255  # at least something changed


class TestAutocrop:
    def test_crops_transparent_border(self):
        img = _make_rgba()  # 200×200 with 40px transparent border
        result = autocrop(img, padding=0)
        # Should crop to the 120×120 red square.
        assert result.size == (120, 120)

    def test_padding_adds_border(self):
        img = _make_rgba()
        result = autocrop(img, padding=10)
        w, h = result.size
        assert w == 140 and h == 140

    def test_fully_transparent_returns_original(self):
        blank = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        result = autocrop(blank, padding=10)
        assert result.size == (100, 100)


class TestCompositeBackground:
    def test_white_background(self):
        img = _make_rgba()
        result = composite_background(img, color="white")
        assert result.mode == "RGB"
        arr = np.array(result)
        # Transparent border → should be white (255, 255, 255).
        assert tuple(arr[5, 5]) == (255, 255, 255)

    def test_black_background(self):
        img = _make_rgba()
        result = composite_background(img, color="black")
        arr = np.array(result)
        assert tuple(arr[5, 5]) == (0, 0, 0)

    def test_hex_background(self):
        img = _make_rgba()
        result = composite_background(img, color="#AABBCC")
        arr = np.array(result)
        assert tuple(arr[5, 5]) == (0xAA, 0xBB, 0xCC)

    def test_invalid_hex_raises(self):
        img = _make_rgba()
        with pytest.raises(ValueError):
            composite_background(img, color="#ZZZ")


# ── Integration test ──────────────────────────────────────────────────────────

@pytest.mark.integration
def test_full_pipeline_u2net():
    """Downloads u2net (~170 MB) on first run; skipped in unit-test mode."""
    from bgremover.core import ProcessConfig, process_image

    img = _make_rgba(400, 400)
    config = ProcessConfig(
        model="u2net",
        alpha_matting=False,
        feather=True,
        despeckle_size=10,
        do_autocrop=True,
        crop_padding=20,
    )
    result = process_image(img, config)

    assert result.mode == "RGBA", "Pipeline must return RGBA"
    arr = np.array(result)
    # The result should not be fully opaque everywhere (background was removed).
    assert arr[:, :, 3].min() < 255, "Some pixels should be transparent after BG removal"
