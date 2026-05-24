#!/usr/bin/env python3
"""Gradio web UI for the jewelry background-removal pipeline."""
from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps

from bgremover.core import PRESETS, SUPPORTED_MODELS, ProcessConfig, process_image

PRESET_NAMES = ["(none)"] + list(PRESETS.keys())
BG_CHOICES = ["transparent", "white", "black", "custom hex"]
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tiff", ".tif"}


def _to_path(f) -> str:
    return f if isinstance(f, str) else f.name


def update_from_preset(preset_name: str):
    if preset_name == "(none)" or preset_name not in PRESETS:
        return [gr.update()] * 10
    p = PRESETS[preset_name]
    return [
        gr.update(value=p["model"]),
        gr.update(value=p["alpha_matting"]),
        gr.update(value=p["alpha_matting_fg_threshold"]),
        gr.update(value=p["alpha_matting_bg_threshold"]),
        gr.update(value=p["alpha_matting_erode_size"]),
        gr.update(value=p["feather"]),
        gr.update(value=p["feather_radius"]),
        gr.update(value=p["despeckle_size"]),
        gr.update(value=p["do_autocrop"]),
        gr.update(value=p["crop_padding"]),
    ]


def process_images(
    files_input,
    folder_input,
    preset,
    model,
    alpha_matting,
    fg_threshold,
    bg_threshold,
    erode_size,
    feather,
    feather_radius,
    despeckle_size,
    do_autocrop,
    crop_padding,
    bg_choice,
    custom_hex,
    progress=gr.Progress(track_tqdm=True),
):
    # Collect all uploaded paths from both sources
    all_files: list[str] = []
    if files_input:
        all_files.extend(_to_path(f) for f in files_input)
    if folder_input:
        all_files.extend(_to_path(f) for f in folder_input)

    # Keep only supported image types
    all_files = [f for f in all_files if Path(f).suffix.lower() in SUPPORTED_EXTS]
    if not all_files:
        raise gr.Error("No supported image files found. Please upload JPG, PNG, HEIC, WebP, or TIFF files.")

    # Build ProcessConfig
    if preset and preset != "(none)" and preset in PRESETS:
        cfg = ProcessConfig.from_preset(preset)
    else:
        cfg = ProcessConfig()

    bg_map = {"transparent": None, "white": "white", "black": "black"}
    bg = bg_map.get(bg_choice)
    if bg_choice == "custom hex":
        bg = custom_hex.strip() if custom_hex else None

    cfg = cfg.apply_overrides(
        model=model,
        alpha_matting=alpha_matting,
        alpha_matting_fg_threshold=int(fg_threshold),
        alpha_matting_bg_threshold=int(bg_threshold),
        alpha_matting_erode_size=int(erode_size),
        feather=feather,
        feather_radius=float(feather_radius),
        despeckle_size=int(despeckle_size),
        do_autocrop=do_autocrop,
        crop_padding=int(crop_padding),
        background=bg,
    )

    tmp_dir = tempfile.mkdtemp(prefix="jewelry_nobg_")
    out_paths: list[str] = []
    gallery_items: list[tuple[str, str]] = []
    errors: list[str] = []
    total = len(all_files)

    for i, fp in enumerate(all_files):
        name = Path(fp).name
        progress(i / total, desc=f"[{i + 1}/{total}] {name}")
        try:
            img = Image.open(fp)
            img = ImageOps.exif_transpose(img).convert("RGBA")
            result = process_image(img, cfg)

            out_name = Path(fp).stem + "_nobg.png"
            out_path = os.path.join(tmp_dir, out_name)
            result.save(out_path, format="PNG")

            out_paths.append(out_path)
            gallery_items.append((out_path, out_name))
        except Exception as exc:
            errors.append(f"❌ {name}: {exc}")

    progress(1.0, desc="Done!")

    # Build ZIP archive
    zip_path: str | None = None
    if out_paths:
        zip_path = os.path.join(tmp_dir, "processed_images.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in out_paths:
                zf.write(p, os.path.basename(p))

    lines = [f"✅ {len(out_paths)}/{total} image(s) processed successfully."]
    if errors:
        lines += [""] + errors
    status = "\n".join(lines)

    return gallery_items, out_paths if out_paths else None, zip_path, status


# ── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
#settings-col {
    background: var(--background-fill-secondary);
    border-radius: 12px;
    padding: 10px 14px;
}
.sec-hdr {
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--body-text-color-subdued);
    margin: 14px 0 4px;
    border-bottom: 1px solid var(--border-color-primary);
    padding-bottom: 3px;
}
#process-btn { font-size: 1.05rem; margin-top: 6px; }
#status-box textarea { font-family: monospace; font-size: 0.85rem; }
"""

# ── Build UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Jewelry BG Remover") as demo:
    gr.Markdown(
        "# 💎 Jewelry Background Remover\n"
        "Upload jewelry photos, tune every parameter, and download transparent PNGs."
    )

    with gr.Row(equal_height=False):

        # ── Left: settings ───────────────────────────────────────────────
        with gr.Column(scale=1, min_width=290, elem_id="settings-col"):

            gr.Markdown("## ⚙️ Settings")

            preset = gr.Dropdown(
                choices=PRESET_NAMES,
                value="(none)",
                label="Quick Preset",
                info="Auto-fills every field below; you can still tweak individually",
            )
            model = gr.Dropdown(
                choices=SUPPORTED_MODELS,
                value="birefnet-general",
                label="Model",
                info="birefnet-general = best quality  ·  u2net = fastest  ·  sam = structure-aware",
            )

            gr.HTML("<div class='sec-hdr'>Alpha Matting</div>")
            alpha_matting = gr.Checkbox(value=True, label="Enable alpha matting (refines hair-thin edges)")
            fg_threshold = gr.Slider(0, 255, value=240, step=1, label="Foreground threshold")
            bg_threshold = gr.Slider(0, 255, value=10, step=1, label="Background threshold")
            erode_size = gr.Slider(0, 40, value=10, step=1, label="Erode size (px)")

            gr.HTML("<div class='sec-hdr'>Post-Processing</div>")
            feather = gr.Checkbox(value=True, label="Feather edges")
            feather_radius = gr.Slider(0.0, 10.0, value=1.5, step=0.5, label="Feather radius (px)")
            despeckle_size = gr.Slider(
                0, 500, value=50, step=10,
                label="Despeckle — drop blobs smaller than (px²)",
                info="Removes stray reflections / dust blobs",
            )
            do_autocrop = gr.Checkbox(value=True, label="Auto-crop to content bounding box")
            crop_padding = gr.Slider(0, 200, value=40, step=5, label="Crop padding (px)")

            gr.HTML("<div class='sec-hdr'>Output Background</div>")
            bg_choice = gr.Radio(
                choices=BG_CHOICES,
                value="transparent",
                label=None,
                info="Transparent = RGBA PNG with alpha channel",
            )
            custom_hex = gr.Textbox(
                value="#FFFFFF",
                label="Custom background color (hex)",
                placeholder="#RRGGBB",
                visible=False,
            )

        # ── Right: upload + results ──────────────────────────────────────
        with gr.Column(scale=2):

            gr.Markdown("## 📤 Upload Images")
            with gr.Tabs():
                with gr.Tab("📄 Individual Files"):
                    files_input = gr.File(
                        file_count="multiple",
                        file_types=["image", ".heic", ".heif", ".tiff", ".tif"],
                        label="Drag & drop images here, or click to browse",
                        height=150,
                    )
                with gr.Tab("📁 Entire Folder"):
                    folder_input = gr.File(
                        file_count="directory",
                        label="Select a folder — all images inside will be processed",
                        height=150,
                    )

            process_btn = gr.Button(
                "🚀 Remove Backgrounds",
                variant="primary",
                size="lg",
                elem_id="process-btn",
            )

            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                lines=3,
                elem_id="status-box",
            )

            gr.Markdown("## 🖼️ Results")
            gallery = gr.Gallery(
                label="Processed images — click any to preview at full size",
                columns=3,
                height=420,
                object_fit="contain",
                allow_preview=True,
                buttons=["download", "fullscreen"],
                type="filepath",
            )

            gr.Markdown("### ⬇️ Download")
            with gr.Row():
                download_files = gr.File(
                    file_count="multiple",
                    label="Individual PNG files",
                    interactive=False,
                    height=110,
                )
                download_zip = gr.File(
                    label="All images as ZIP",
                    interactive=False,
                    height=110,
                )

    # ── Event wiring ─────────────────────────────────────────────────────

    all_setting_outputs = [
        model, alpha_matting,
        fg_threshold, bg_threshold, erode_size,
        feather, feather_radius, despeckle_size,
        do_autocrop, crop_padding,
    ]

    preset.change(fn=update_from_preset, inputs=preset, outputs=all_setting_outputs)

    alpha_matting.change(
        fn=lambda x: [gr.update(interactive=x)] * 3,
        inputs=alpha_matting,
        outputs=[fg_threshold, bg_threshold, erode_size],
    )
    feather.change(
        fn=lambda x: gr.update(interactive=x),
        inputs=feather,
        outputs=feather_radius,
    )
    do_autocrop.change(
        fn=lambda x: gr.update(interactive=x),
        inputs=do_autocrop,
        outputs=crop_padding,
    )
    bg_choice.change(
        fn=lambda c: gr.update(visible=c == "custom hex"),
        inputs=bg_choice,
        outputs=custom_hex,
    )

    process_btn.click(
        fn=lambda: gr.update(value="⏳ Processing… check the progress bar above."),
        outputs=status_box,
    ).then(
        fn=process_images,
        inputs=[
            files_input, folder_input,
            preset, model,
            alpha_matting, fg_threshold, bg_threshold, erode_size,
            feather, feather_radius, despeckle_size,
            do_autocrop, crop_padding,
            bg_choice, custom_hex,
        ],
        outputs=[gallery, download_files, download_zip, status_box],
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft(), css=CSS)
