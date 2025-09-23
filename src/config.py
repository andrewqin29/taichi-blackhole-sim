from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Tuple


@dataclass
class RenderSettings:
    resolution: Tuple[int, int] = (1280, 720)
    exposure: float = -0.25
    gamma: float = 2.2


@dataclass
class CameraSettings:
    position: Tuple[float, float, float] = (0.0, -22.0, 2.5)
    forward: Tuple[float, float, float] = (0.0, 0.993, -0.118)
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    vfov_degrees: float = 55.0
    near_plane: float = 0.1
    far_plane: float = 120.0


@dataclass
class BlackHoleSettings:
    mass: float = 1.0
    horizon_epsilon: float = 5e-4
    disk_inner_radius: float = 6.0
    disk_outer_radius: float = 45.0
    disk_inclination_degrees: float = 18.0


@dataclass
class IntegrationSettings:
    step_size: float = 0.015
    max_steps: int = 1400
    escape_radius: float = 220.0


@dataclass
class ControllerSettings:
    move_speed: float = 0.8
    boost_multiplier: float = 3.0
    mouse_sensitivity: float = 0.0022


@dataclass
class OutputSettings:
    screenshot_directory: Path = Path.home() / "Desktop"
    filename_prefix: str = "blackhole"


@dataclass
class SimulationConfig:
    render: RenderSettings
    camera: CameraSettings
    black_hole: BlackHoleSettings
    integration: IntegrationSettings
    controller: ControllerSettings
    output: OutputSettings


def build_reference_config() -> SimulationConfig:
    return SimulationConfig(
        render=RenderSettings(),
        camera=CameraSettings(),
        black_hole=BlackHoleSettings(),
        integration=IntegrationSettings(),
        controller=ControllerSettings(),
        output=OutputSettings(),
    )


def build_development_config(*, resolution_scale: float = 0.5) -> SimulationConfig:
    if resolution_scale <= 0.0:
        raise ValueError("resolution_scale must be positive")

    base = build_reference_config()
    scaled_resolution = tuple(max(1, int(round(dim * resolution_scale))) for dim in base.render.resolution)
    tweaked_render = replace(base.render, resolution=scaled_resolution)
    tweaked_integration = replace(
        base.integration,
        max_steps=max(1, int(base.integration.max_steps * resolution_scale)),
    )

    return SimulationConfig(
        render=tweaked_render,
        camera=replace(base.camera),
        black_hole=replace(base.black_hole),
        integration=tweaked_integration,
        controller=replace(base.controller),
        output=replace(base.output),
    )


def validate_config(config: SimulationConfig) -> None:
    width, height = config.render.resolution
    if width <= 0 or height <= 0:
        raise ValueError("resolution must be positive")

    if config.render.gamma <= 0.0:
        raise ValueError("gamma must be positive")

    if config.integration.step_size <= 0.0:
        raise ValueError("step size must be positive")

    if config.integration.max_steps <= 0:
        raise ValueError("max_steps must be positive")

    if config.integration.escape_radius <= 0.0:
        raise ValueError("escape radius must be positive")

    if not (0.0 < config.camera.vfov_degrees < 180.0):
        raise ValueError("camera vfov must be between 0 and 180")

    if config.camera.forward == (0.0, 0.0, 0.0):
        raise ValueError("camera forward cannot be zero")

    if config.black_hole.disk_inner_radius <= 0.0:
        raise ValueError("disk inner radius must be positive")

    if config.black_hole.disk_outer_radius <= config.black_hole.disk_inner_radius:
        raise ValueError("disk outer radius must exceed inner radius")

    if config.black_hole.horizon_epsilon <= 0.0:
        raise ValueError("horizon epsilon must be positive")

    if config.controller.move_speed <= 0.0:
        raise ValueError("controller move_speed must be positive")

    if config.controller.mouse_sensitivity <= 0.0:
        raise ValueError("mouse sensitivity must be positive")

    screenshot_dir = config.output.screenshot_directory.expanduser()
    if screenshot_dir.exists() and not screenshot_dir.is_dir():
        raise ValueError("screenshot path collides with a non-directory")


