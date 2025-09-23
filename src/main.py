"""Program entry point: boots Taichi, runs the renderer, and handles GUI controls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import taichi as ti

try:  # support "python -m src.main" and "python src/main.py"
    if __package__ is None or __package__ == "":
        _ROOT = Path(__file__).resolve().parent.parent
        if str(_ROOT) not in sys.path:
            sys.path.append(str(_ROOT))
        from src.config import (  # type: ignore
            build_development_config,
            build_reference_config,
            validate_config,
        )
        from src.kernel import Renderer
    else:  # pragma: no cover - exercised in package mode
        from .config import build_development_config, build_reference_config, validate_config
        from .kernel import Renderer
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("failed to import project modules") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relativistic black hole renderer")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="run with reduced resolution/steps for quick iteration",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="resolution scale factor when --dev is supplied (default: 0.5)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="force Taichi to use the CPU backend",
    )
    return parser.parse_args()


def _select_arch(force_cpu: bool) -> int:
    if force_cpu:
        return ti.cpu
    if ti.vulkan_available():
        return ti.vulkan
    if ti.gpu_available():
        return ti.gpu
    return ti.cpu


def _with_config(args: argparse.Namespace):
    builder: Callable[..., object]
    if args.dev:
        builder = build_development_config
        config = builder(resolution_scale=args.scale)
    else:
        builder = build_reference_config
        config = builder()
    validate_config(config)
    return config


def _ensure_output_path(path: Path) -> Path:
    expanded = path.expanduser()
    expanded.mkdir(parents=True, exist_ok=True)
    return expanded


def main() -> None:
    args = _parse_args()
    config = _with_config(args)

    ti.init(arch=_select_arch(args.cpu), offline_cache=True)

    renderer = Renderer(config)
    width, height = config.render.resolution
    window = ti.ui.Window("Black Hole", (width, height), vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()

    screenshot_dir = _ensure_output_path(config.output.screenshot_directory)
    frame_id = 0

    while window.running:
        for event in window.get_events(ti.ui.PRESS):
            if event.key in (ti.ui.ESCAPE, ti.ui.QUIT):
                window.running = False
            elif event.key == "s":
                rgb = renderer.readback()
                image = np.transpose(rgb, (1, 0, 2))
                filename = f"{config.output.filename_prefix}_{frame_id:04d}.png"
                ti.tools.imwrite(image, str(screenshot_dir / filename))
            elif event.key == "[":
                renderer.adjust_exposure(-0.1)
            elif event.key == "]":
                renderer.adjust_exposure(0.1)

        renderer.render()
        canvas.set_image(renderer.targets["color"])

        gui.text(f"Exposure: {renderer.exposure[None]:.2f}")
        gui.text("Press S to save a screenshot")
        gui.text("Use [ / ] to tweak exposure")
        gui.text("Esc quits")
        window.show()

        frame_id += 1


if __name__ == "__main__":
    main()

