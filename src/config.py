"""Configuration scaffolding for the Taichi black hole renderer.

This module gathers all tunable parameters for the simulation, rendering,
and interaction layers. Treat it as the single source of truth that other
modules query when they need domain knowledge (camera pose, integration
steps, physical constants, controller speeds, output locations, etc.).

Guidance for completing this module:
- Flesh out the dataclasses with the exact parameters once you start
  wiring up the downstream modules.
- Keep values physically meaningful where appropriate; aim for the
  reference preset to reproduce the original visual target.
- Consider adding helper presets (e.g., debug, low-res) once the
  reference preset works.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class RenderSettings:
    """Minimal structure for resolution and tonemapping parameters."""

    # TODO: add render-stage parameters (exposure, gamma) when kernel is ready.
    resolution: Tuple[int, int] = (960, 540)
    exposure: float = 0.0
    gamma: float = 2.2


@dataclass
class CameraSettings:
    """Holds camera origin, orientation, and integration controls."""

    # TODO: populate with actual pose values once the camera math is in place.
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    forward: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    vfov_degrees: float = 60.0
    near_plane: float = 0.1
    far_plane: float = 1000.0


@dataclass
class BlackHoleSettings:
    """Stores mass, horizon epsilon, and disk geometry for the scene."""

    # TODO: refine these once the physics layer defines the unit system.
    mass: float = 1.0
    horizon_epsilon: float = 1e-3
    disk_inner_radius: float = 6.0
    disk_outer_radius: float = 60.0
    disk_inclination_degrees: float = 0.0


@dataclass
class IntegrationSettings:
    """Collects RK4 step size, max steps, and termination thresholds."""

    # TODO: harmonize these with the geodesic integrator.
    step_size: float = 0.01
    max_steps: int = 1024
    escape_radius: float = 500.0


@dataclass
class ControllerSettings:
    """FPS-style controller speeds and sensitivity defaults."""

    # TODO: tune movement speed and mouse sensitivity after basic camera control works.
    move_speed: float = 1.0
    boost_multiplier: float = 3.0
    mouse_sensitivity: float = 0.0025


@dataclass
class OutputSettings:
    """Paths and naming conventions for saved frames."""

    # TODO: when implementing frame capture, ensure the path exists per platform.
    screenshot_directory: Path = Path.home() / "Desktop"
    filename_prefix: str = "blackhole"


@dataclass
class SimulationConfig:
    """Aggregated configuration object passed across modules."""

    render: RenderSettings
    camera: CameraSettings
    black_hole: BlackHoleSettings
    integration: IntegrationSettings
    controller: ControllerSettings
    output: OutputSettings


# build_reference_config will produce the canonical parameter set used to
# match the original project's visuals. main.py will call this at startup and
# distribute the result to modules like scene.py, physics.py, and kernel.py.
def build_reference_config() -> SimulationConfig:
    """Implementation outline:
    1. Instantiate each settings dataclass with tuned values.
    2. Adjust parameters (e.g., disk inclination, camera pose) until the
       rendered frame aligns with the reference look.
    3. Return a SimulationConfig bundling all sections.
    """
    raise NotImplementedError("Fill in the reference parameter values once ready.")


# build_development_config allows you to derive alternate presets (e.g.,
# low-res previews) from the reference. Other modules should treat this as
# optional sugar and always accept a SimulationConfig in their APIs.
def build_development_config(*, resolution_scale: float = 0.5) -> SimulationConfig:
    """Implementation outline:
    1. Start from build_reference_config once it is implemented.
    2. Adjust the render resolution and any integrator shortcuts needed for
       quick iteration.
    3. Return the tweaked SimulationConfig.
    """
    raise NotImplementedError("Derive a development preset after reference config exists.")


# validate_config should run lightweight assertions to catch typos or unit
# mismatches early. Call it from whichever module first consumes the config
# (likely main.py) before the simulation starts.
def validate_config(config: SimulationConfig) -> None:
    """Implementation outline:
    1. Assert positive resolutions, step sizes, and radii.
    2. Ensure disk_inner_radius < disk_outer_radius and the screenshot path looks valid.
    3. Extend with more checks as the project grows.
    """
    raise NotImplementedError("Add sanity checks when SimulationConfig gains real values.")
