#!/usr/bin/env python3
"""CLI entry point for the jewelry background-removal pipeline."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Optional

import sys
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Force UTF-8 output on Windows so Rich spinners don't crash on narrow codepages.
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from tqdm import tqdm

from bgremover.core import PRESETS, SUPPORTED_MODELS, ProcessConfig, process_image
from bgremover.io_utils import iter_images, load_image, output_path, save_image

app = typer.Typer(
    name="remove-bg",
    help="Remove backgrounds from jewelry product photos.",
    pretty_exceptions_show_locals=False,
)
console = Console()


def _build_config(
    preset: Optional[str],
    model: Optional[str],
    alpha_matting: Optional[bool],
    fg_threshold: Optional[int],
    bg_threshold: Optional[int],
    erode_size: Optional[int],
    feather: Optional[bool],
    feather_radius: Optional[float],
    despeckle_size: Optional[int],
    do_autocrop: Optional[bool],
    crop_padding: Optional[int],
    background: Optional[str],
) -> ProcessConfig:
    """Build ProcessConfig: preset first, then explicit overrides."""
    if preset:
        cfg = ProcessConfig.from_preset(preset)
    else:
        cfg = ProcessConfig()  # defaults: birefnet-general, alpha matting on, feather on

    overrides = {
        "model": model,
        "alpha_matting": alpha_matting,
        "alpha_matting_fg_threshold": fg_threshold,
        "alpha_matting_bg_threshold": bg_threshold,
        "alpha_matting_erode_size": erode_size,
        "feather": feather,
        "feather_radius": feather_radius,
        "despeckle_size": despeckle_size,
        "do_autocrop": do_autocrop,
        "crop_padding": crop_padding,
        "background": background,
    }
    return cfg.apply_overrides(**overrides)


@app.command()
def main(
    input: Annotated[Path, typer.Argument(help="Input image file or folder (with --batch).")],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file (single) or folder (batch)."),
    ] = None,
    # ── Mode ──────────────────────────────────────────────────────────────────
    batch: Annotated[
        bool,
        typer.Option("--batch", "-b", help="Process every image in the input folder."),
    ] = False,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Parallel workers for batch mode."),
    ] = 4,
    # ── Preset / model ────────────────────────────────────────────────────────
    preset: Annotated[
        Optional[str],
        typer.Option(
            "--preset",
            "-p",
            help=f"Quality preset. One of: {list(PRESETS)}. Overrides per-flag defaults.",
        ),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help=f"Segmentation model. One of: {SUPPORTED_MODELS}. Default: birefnet-general.",
        ),
    ] = None,
    # ── Alpha matting ─────────────────────────────────────────────────────────
    alpha_matting: Annotated[
        Optional[bool],
        typer.Option(
            "--alpha-matting/--no-alpha-matting",
            help="Refine edges with alpha matting (slower but cleaner).",
        ),
    ] = None,
    fg_threshold: Annotated[
        Optional[int],
        typer.Option("--fg-threshold", help="Alpha matting foreground threshold (0–255)."),
    ] = None,
    bg_threshold: Annotated[
        Optional[int],
        typer.Option("--bg-threshold", help="Alpha matting background threshold (0–255)."),
    ] = None,
    erode_size: Annotated[
        Optional[int],
        typer.Option("--erode-size", help="Alpha matting erode kernel size in pixels."),
    ] = None,
    # ── Post-processing ───────────────────────────────────────────────────────
    feather: Annotated[
        Optional[bool],
        typer.Option("--feather/--no-feather", help="Blur alpha edges for a soft look."),
    ] = None,
    feather_radius: Annotated[
        Optional[float],
        typer.Option("--feather-radius", help="Gaussian blur radius for edge feathering."),
    ] = None,
    despeckle_size: Annotated[
        Optional[int],
        typer.Option("--despeckle", help="Drop alpha components smaller than N pixels."),
    ] = None,
    do_autocrop: Annotated[
        Optional[bool],
        typer.Option("--autocrop/--no-autocrop", help="Crop to content bounding box."),
    ] = None,
    crop_padding: Annotated[
        Optional[int],
        typer.Option("--padding", help="Pixels of padding around the auto-crop box."),
    ] = None,
    background: Annotated[
        Optional[str],
        typer.Option(
            "--background",
            help="Composite onto 'white', 'black', or a hex color like '#F5F5F5'. "
            "Omit for a transparent PNG.",
        ),
    ] = None,
) -> None:
    config = _build_config(
        preset, model, alpha_matting, fg_threshold, bg_threshold,
        erode_size, feather, feather_radius, despeckle_size,
        do_autocrop, crop_padding, background,
    )

    if batch or input.is_dir():
        _run_batch(input, output, config, workers)
    else:
        _run_single(input, output, config)


def _run_single(input_path: Path, out: Optional[Path], config: ProcessConfig) -> None:
    if not input_path.is_file():
        console.print(f"[red]Error:[/red] '{input_path}' is not a file. Use --batch for folders.")
        raise typer.Exit(1)

    dest = out if out else output_path(input_path, None)

    console.print(
        f"[cyan]Model:[/cyan] {config.model}  "
        f"[cyan]Alpha matting:[/cyan] {config.alpha_matting}  "
        f"[cyan]Feather:[/cyan] {config.feather}"
    )

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task(f"Processing {input_path.name} …", total=None)
        image = load_image(input_path)
        result = process_image(image, config)
        save_image(result, dest)
        progress.update(task, completed=1)

    console.print(f"[green]Saved:[/green] {dest}")


def _process_one(args: tuple) -> tuple[Path, Optional[str]]:
    """Worker helper for batch mode. Returns (output_path, error_message|None)."""
    in_path, out_dir, config = args
    try:
        dest = output_path(in_path, out_dir)
        img = load_image(in_path)
        result = process_image(img, config)
        save_image(result, dest)
        return dest, None
    except Exception as exc:  # noqa: BLE001
        return in_path, str(exc)


def _run_batch(folder: Path, out_dir: Optional[Path], config: ProcessConfig, workers: int) -> None:
    if not folder.is_dir():
        console.print(f"[red]Error:[/red] '{folder}' is not a directory.")
        raise typer.Exit(1)

    paths = list(iter_images(folder))
    if not paths:
        console.print(f"[yellow]No supported images found in {folder}[/yellow]")
        return

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[cyan]Batch:[/cyan] {len(paths)} images  "
        f"[cyan]Model:[/cyan] {config.model}  "
        f"[cyan]Workers:[/cyan] {workers}"
    )

    args = [(p, out_dir, config) for p in paths]
    errors: list[tuple[Path, str]] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, a): a[0] for a in args}
        for future in tqdm(as_completed(futures), total=len(futures), unit="img"):
            dest, err = future.result()
            if err:
                errors.append((futures[future], err))
            else:
                tqdm.write(f"  ✓ {dest}")

    if errors:
        console.print(f"\n[red]{len(errors)} error(s):[/red]")
        for path, err in errors:
            console.print(f"  [red]{path.name}:[/red] {err}")
        raise typer.Exit(1)

    console.print(f"\n[green]Done.[/green] {len(paths) - len(errors)}/{len(paths)} images saved.")


if __name__ == "__main__":
    app()
