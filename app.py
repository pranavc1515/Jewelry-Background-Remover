#!/usr/bin/env python3
"""Gradio web UI for jewelry background removal.

Launch with:  python app.py
Then open:    http://localhost:7860
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr
from PIL import Image

from bgremover.core import PRESETS, SUPPORTED_MODELS, ProcessConfig, process_image

# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_config_from_ui(
    preset: str,
    model: str,
    alpha_matting: bool,
    fg_threshold: int,
    bg_threshold: int,
    erode_size: int,
    feather: bool,
    feather_radius: float,
    despeckle_size: int,
    do_autocrop: bool,
    crop_padding: int,
    background: str,
) -> ProcessConfig:
    if preset != "custom":
        cfg = ProcessConfig.from_preset(preset)
    else:
        cfg = ProcessConfig(
            model=model,
            alpha_matting=alpha_matting,
            alpha_matting_fg_threshold=int(fg_threshold),
            alpha_matting_bg_threshold=int(bg_threshold),
            alpha_matting_erode_size=int(erode_size),
            feather=feather,
            feather_radius=feather_radius,
            despeckle_size=int(despeckle_size),
            do_autocrop=do_autocrop,
            crop_padding=int(crop_padding),
            background=background.strip() or None,
        )
    # Always honour explicit model choice from the dropdown.
    if model and model != PRESETS.get(preset, {}).get("model"):
        cfg.model = model
    return cfg


def _ui_defaults_from_preset(preset: str) -> tuple:
    """Return UI control values matching the chosen preset."""
    if preset == "custom":
        p = PRESETS["balanced"]
    else:
        p = PRESETS[preset]
    return (
        p["model"],
        p["alpha_matting"],
        p["alpha_matting_fg_threshold"],
        p["alpha_matting_bg_threshold"],
        p["alpha_matting_erode_size"],
        p["feather"],
        p["feather_radius"],
        p["despeckle_size"],
        p["do_autocrop"],
        p["crop_padding"],
    )


def run_removal(
    input_image: Optional[Image.Image],
    preset: str,
    model: str,
    alpha_matting: bool,
    fg_threshold: int,
    bg_threshold: int,
    erode_size: int,
    feather: bool,
    feather_radius: float,
    despeckle_size: int,
    do_autocrop: bool,
    crop_padding: int,
    background: str,
) -> tuple[Optional[Image.Image], Optional[str]]:
    if input_image is None:
        return None, None

    if input_image.mode != "RGBA":
        input_image = input_image.convert("RGBA")

    config = _build_config_from_ui(
        preset, model, alpha_matting, fg_threshold, bg_threshold,
        erode_size, feather, feather_radius, despeckle_size,
        do_autocrop, crop_padding, background,
    )

    result = process_image(input_image, config)

    # Write to a temp file so Gradio can offer a download link.
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    result.save(tmp.name, format="PNG")

    return result, tmp.name


# ── Layout ────────────────────────────────────────────────────────────────────

PRESET_CHOICES = ["best", "balanced", "fast", "custom"]

with gr.Blocks(title="Jewelry Background Remover") as demo:
    gr.Markdown(
        "# Jewelry Background Remover\n"
        "Upload a jewelry photo to extract a clean transparent PNG. "
        "Fine details — pearls, chains, rhinestones — are preserved best with the **best** preset."
    )

    with gr.Row():
        # ── Left column: input + controls ──────────────────────────────────
        with gr.Column(scale=1, min_width=320):
            input_image = gr.Image(
                label="Upload Image (JPG / PNG / HEIC)",
                type="pil",
                sources=["upload", "clipboard"],
            )

            preset_dd = gr.Dropdown(
                choices=PRESET_CHOICES,
                value="best",
                label="Preset",
            )
            model_dd = gr.Dropdown(
                choices=SUPPORTED_MODELS,
                value="birefnet-general",
                label="Model",
            )

            with gr.Accordion("Advanced Settings", open=False):
                alpha_matting_cb = gr.Checkbox(value=True, label="Alpha Matting")
                with gr.Row():
                    fg_slider = gr.Slider(0, 255, value=240, step=1, label="FG Threshold")
                    bg_slider = gr.Slider(0, 255, value=10, step=1, label="BG Threshold")
                erode_slider = gr.Slider(0, 30, value=10, step=1, label="Erode Size (px)")

                gr.Markdown("---")
                feather_cb = gr.Checkbox(value=True, label="Edge Feathering")
                feather_r_slider = gr.Slider(0.0, 8.0, value=1.5, step=0.5, label="Feather Radius")

                gr.Markdown("---")
                despeckle_slider = gr.Slider(0, 1000, value=50, step=10, label="Despeckle Min Size (px)")

                gr.Markdown("---")
                autocrop_cb = gr.Checkbox(value=True, label="Auto-Crop")
                padding_slider = gr.Slider(0, 200, value=40, step=5, label="Crop Padding (px)")

                gr.Markdown("---")
                bg_color_tb = gr.Textbox(
                    value="",
                    placeholder="white / black / #F5F5F5  (empty = transparent PNG)",
                    label="Background Color",
                )

            run_btn = gr.Button("Remove Background", variant="primary", size="lg")

        # ── Right column: before / after ───────────────────────────────────
        with gr.Column(scale=1, min_width=320):
            with gr.Row():
                before_img = gr.Image(label="Original", type="pil", interactive=False)
                after_img = gr.Image(label="Result", type="pil", interactive=False)
            download_file = gr.File(label="Download transparent PNG")

    # ── Event wiring ─────────────────────────────────────────────────────────

    # Mirror uploaded image to the "before" panel immediately.
    input_image.change(lambda img: img, inputs=input_image, outputs=before_img)

    # When preset changes, propagate defaults to all advanced controls.
    def on_preset_change(preset: str):
        vals = _ui_defaults_from_preset(preset)
        return vals  # must match the 10-element list below

    preset_dd.change(
        on_preset_change,
        inputs=preset_dd,
        outputs=[
            model_dd, alpha_matting_cb,
            fg_slider, bg_slider, erode_slider,
            feather_cb, feather_r_slider,
            despeckle_slider, autocrop_cb, padding_slider,
        ],
    )

    run_btn.click(
        run_removal,
        inputs=[
            input_image, preset_dd, model_dd,
            alpha_matting_cb, fg_slider, bg_slider, erode_slider,
            feather_cb, feather_r_slider, despeckle_slider,
            autocrop_cb, padding_slider, bg_color_tb,
        ],
        outputs=[after_img, download_file],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
