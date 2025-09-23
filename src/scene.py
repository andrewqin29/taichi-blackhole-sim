"""Scene assembly and ray setup for the Taichi black hole renderer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple

from .config import SimulationConfig
from .physics import GeodesicState


Vector3 = Tuple[float, float, float]
Vector4 = Tuple[float, float, float, float]


@dataclass
class CameraFrame:
    """FPS-style camera basis derived from config camera settings."""

    origin: Vector3
    forward: Vector3
    right: Vector3
    up: Vector3


@dataclass
class RaySample:
    """Initial conditions for a geodesic launched through a pixel."""

    state: GeodesicState
    pixel: Tuple[int, int]


def _normalize(vec: Vector3) -> Vector3:
    # prevent division by zero before normalization
    length = math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
    if length <= 0.0:
        raise ValueError("cannot normalize zero-length vector")
    return (vec[0] / length, vec[1] / length, vec[2] / length)


def _cross(a: Vector3, b: Vector3) -> Vector3:
    # compute vector product for orthonormal frame construction
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def build_camera_frame(config: SimulationConfig) -> CameraFrame:
    """Derive an orthonormal camera basis aligned with the config orientation."""

    forward = _normalize(config.camera.forward)
    up = _normalize(config.camera.up)
    right = _normalize(_cross(forward, up))
    corrected_up = _cross(right, forward)  # in case forward and up are not initially perfectly orthogonal

    return CameraFrame(
        origin=config.camera.position,
        forward=forward,
        right=right,
        up=_normalize(corrected_up),
    )


def _pixel_to_ndc(pixel: Tuple[int, int], resolution: Tuple[int, int]) -> Tuple[float, float]:
    # map integer pixel index to normalized device coordinates (-1..1)
    x, y = pixel
    width, height = resolution
    ndc_x = (2.0 * ((x + 0.5) / width)) - 1.0
    ndc_y = 1.0 - (2.0 * ((y + 0.5) / height))
    return ndc_x, ndc_y


def _ndc_to_camera_dir(
    ndc: Tuple[float, float],
    frame: CameraFrame,
    vfov_degrees: float,
    resolution: Tuple[int, int],
) -> Vector3:
    # convert ndc coordinates into a direction in camera space with aspect compensation
    ndc_x, ndc_y = ndc
    width, height = resolution
    if height <= 0:
        raise ValueError("resolution height must be positive")
    aspect = width / height
    tan_half_fov = math.tan(math.radians(vfov_degrees) * 0.5)
    dir_camera = (
        frame.forward[0]
        + ndc_x * tan_half_fov * aspect * frame.right[0]
        + ndc_y * tan_half_fov * frame.up[0],
        frame.forward[1]
        + ndc_x * tan_half_fov * aspect * frame.right[1]
        + ndc_y * tan_half_fov * frame.up[1],
        frame.forward[2]
        + ndc_x * tan_half_fov * aspect * frame.right[2]
        + ndc_y * tan_half_fov * frame.up[2],
    )
    return _normalize(dir_camera)


def _cartesian_to_spherical(origin: Vector3, direction: Vector3, mass: float) -> Tuple[Vector4, Vector4]:
    # convert ray origin and direction into schwarzschild coordinates
    ox, oy, oz = origin
    dx, dy, dz = direction

    r = math.sqrt(ox * ox + oy * oy + oz * oz)
    theta = math.acos(oz / r) if r > 0.0 else math.pi * 0.5
    phi = math.atan2(oy, ox)

    # approximate null momentum with unit affine parameter scaling
    dt = 1.0
    dr = dx
    dtheta = dy / max(r, 1e-6)
    dphi = dz / max(r * math.sin(theta), 1e-6)

    position = (0.0, r, theta, phi)
    momentum = (dt, dr, dtheta, dphi)

    return position, momentum


def generate_primary_geodesic(
    pixel: Tuple[int, int],
    config: SimulationConfig,
    frame: CameraFrame,
) -> RaySample:
    """Map pixel coordinates to an initial geodesic state launching into the scene."""

    ndc = _pixel_to_ndc(pixel, config.render.resolution)
    direction = _ndc_to_camera_dir(ndc, frame, config.camera.vfov_degrees, config.render.resolution)
    position, momentum = _cartesian_to_spherical(frame.origin, direction, config.black_hole.mass)
    state = GeodesicState(position=position, momentum=momentum)
    return RaySample(state=state, pixel=pixel)


def horizon_termination_factory(config: SimulationConfig) -> Callable[[GeodesicState], bool]:
    """Produce a closure that reports when a geodesic crosses the horizon."""

    horizon_radius = 2.0 * config.black_hole.mass + config.black_hole.horizon_epsilon

    def condition(state: GeodesicState) -> bool:
        # check radial coordinate against horizon radius
        return state.position[1] <= horizon_radius

    return condition


def escape_condition(config: SimulationConfig, state: GeodesicState) -> bool:
    """Return true if the ray escaped beyond the configured radius."""

    return state.position[1] >= config.integration.escape_radius


def disk_shading_inputs(state: GeodesicState, config: SimulationConfig):
    """Placeholder for gathering disk intersection data to pass into shading."""

    # expose radial coordinate and momentum for future shading calculations
    return {
        "radius": state.position[1],
        "momentum": state.momentum,
        "config": config.black_hole,
    }
