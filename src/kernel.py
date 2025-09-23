"""Taichi render scaffolding: wires config, scene helpers, and physics integrator."""

from __future__ import annotations

import math
from typing import Any, Dict

import taichi as ti

from .config import SimulationConfig
from .scene import CameraFrame, build_camera_frame, horizon_termination_factory

vec3f = ti.types.vector(3, ti.f32)
vec4f = ti.types.vector(4, ti.f32)


@ti.func
def _fract(x: ti.f32) -> ti.f32:
    return x - ti.floor(x)


def make_render_targets(config: SimulationConfig) -> Dict[str, Any]:
    """Allocate the render targets used by the GUI."""

    width, height = config.render.resolution
    color = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
    return {"color": color}


def _cartesian_to_spherical(vec: CameraFrame) -> tuple[float, float, float]:
    x, y, z = vec.origin
    r = math.sqrt(x * x + y * y + z * z)
    if r <= 0.0:
        raise ValueError("camera must be positioned away from the singularity")
    theta = math.acos(max(-1.0, min(1.0, z / r)))
    phi = math.atan2(y, x)
    return r, theta, phi


def push_constants(config: SimulationConfig) -> Dict[str, Any]:
    """Precompute scalar constants that will be uploaded into Taichi fields."""

    mass = config.black_hole.mass
    horizon_radius = 2.0 * mass + config.black_hole.horizon_epsilon
    escape_radius = config.integration.escape_radius

    camera_frame = build_camera_frame(config)
    camera_r, camera_theta, camera_phi = _cartesian_to_spherical(camera_frame)
    lapse_at_camera = 1.0 - (2.0 * mass) / camera_r
    if lapse_at_camera <= 0.0:
        raise ValueError("camera resides inside the event horizon; increase its radius")

    observer_t = 1.0 / math.sqrt(lapse_at_camera)
    disk_inclination = math.radians(config.black_hole.disk_inclination_degrees)
    disk_normal = (
        0.0,
        math.sin(disk_inclination),
        math.cos(disk_inclination),
    )

    width, height = config.render.resolution
    step_size = config.integration.step_size

    return {
        "resolution": (width, height),
        "mass": mass,
        "horizon_radius": horizon_radius,
        "escape_radius": escape_radius,
        "disk_inner": config.black_hole.disk_inner_radius,
        "disk_outer": config.black_hole.disk_outer_radius,
        "disk_inclination": disk_inclination,
        "disk_normal": disk_normal,
        "observer_four_velocity": (observer_t, 0.0, 0.0, 0.0),
        "camera_spherical": (camera_r, camera_theta, camera_phi),
        "plane_epsilon": max(1e-4, step_size * 4.0),
        "step_size": step_size,
        "max_steps": config.integration.max_steps,
        "tan_half_fov": math.tan(math.radians(config.camera.vfov_degrees) * 0.5),
        "aspect": width / height,
    }


@ti.data_oriented
class Renderer:
    """Owns Taichi resources and evaluates the curved-spacetime ray marcher."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.frame = build_camera_frame(config)
        self.constants = push_constants(config)

        self.width, self.height = self.constants["resolution"]
        self.targets = make_render_targets(config)
        self.color = self.targets["color"]

        # upload mutable camera basis so main.py can update it later
        self.origin = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.forward = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.right = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.up = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.camera_spherical = ti.Vector.field(3, dtype=ti.f32, shape=())
        self._upload_camera_frame(self.frame, self.constants["camera_spherical"])

        # scalar uniforms that the kernel pulls from
        self.mass = float(self.constants["mass"])
        self.horizon_radius = float(self.constants["horizon_radius"])
        self.escape_radius = float(self.constants["escape_radius"])
        self.step_size = float(self.constants["step_size"])
        self.max_steps = int(self.constants["max_steps"])
        self.disk_inner = float(self.constants["disk_inner"])
        self.disk_outer = float(self.constants["disk_outer"])
        self.disk_inclination = float(self.constants["disk_inclination"])
        self.disk_sin = math.sin(self.disk_inclination)
        self.disk_cos = math.cos(self.disk_inclination)
        self.plane_epsilon = float(self.constants["plane_epsilon"])
        self.tan_half_fov = float(self.constants["tan_half_fov"])
        self.aspect = float(self.constants["aspect"])

        self.disk_normal = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.disk_normal[None] = tuple(self.constants["disk_normal"])
        self.observer_four_velocity = ti.Vector.field(4, dtype=ti.f32, shape=())
        self.observer_four_velocity[None] = tuple(self.constants["observer_four_velocity"])

        self.theta_min = 1e-4
        self.theta_max = math.pi - 1e-4
        self.grid_density = 22.0
        self.grid_line_width = 140.0

        self.exposure = ti.field(dtype=ti.f32, shape=())
        self.gamma = ti.field(dtype=ti.f32, shape=())
        self.exposure[None] = float(config.render.exposure)
        self.gamma[None] = float(config.render.gamma)

        # retains the CPU-side helper for optional diagnostics outside kernels
        self.horizon_test = horizon_termination_factory(config)

    def _upload_camera_frame(
        self,
        frame: CameraFrame,
        spherical: tuple[float, float, float],
    ) -> None:
        self.origin[None] = tuple(frame.origin)
        self.forward[None] = tuple(frame.forward)
        self.right[None] = tuple(frame.right)
        self.up[None] = tuple(frame.up)
        self.camera_spherical[None] = spherical

    @ti.func
    def _normalize3(self, v: vec3f) -> vec3f:
        length = ti.sqrt(ti.max(v.dot(v), 1e-12))
        return v / length

    @ti.func
    def _metric_diagonal(self, r: ti.f32, theta: ti.f32) -> vec4f:
        mass = ti.static(self.mass)
        safe_r = ti.max(r, 1e-4)
        lapse = 1.0 - 2.0 * mass / safe_r
        lapse = ti.max(lapse, 1e-6)
        sin_theta = ti.sin(theta)
        g_tt = -lapse
        g_rr = 1.0 / lapse
        g_thth = safe_r * safe_r
        g_phph = g_thth * ti.max(sin_theta * sin_theta, 1e-6)
        return vec4f([g_tt, g_rr, g_thth, g_phph])

    @ti.func
    def _covariant_dot(self, diag_metric: vec4f, a: vec4f, b: vec4f) -> ti.f32:
        total = 0.0
        for i in ti.static(range(4)):
            total += diag_metric[i] * a[i] * b[i]
        return total

    @ti.func
    def _emitter_four_velocity(self, radius: ti.f32) -> vec4f:
        mass = ti.static(self.mass)
        safe_r = ti.max(radius, 3.01 * mass)
        denom = ti.max(1.0 - 3.0 * mass / safe_r, 1e-4)
        u_t = 1.0 / ti.sqrt(denom)
        omega = ti.sqrt(mass / (safe_r * safe_r * safe_r))
        return vec4f([u_t, 0.0, 0.0, omega * u_t])

    @ti.func
    def _geodesic_rhs(self, pos: vec4f, mom: vec4f) -> tuple[vec4f, vec4f]:
        r = pos[1]
        theta = pos[2]
        mass = ti.static(self.mass)
        safe_r = ti.max(r, 1e-4)
        lapse = 1.0 - 2.0 * mass / safe_r
        lapse = ti.max(lapse, 1e-6)
        dlapse_dr = 2.0 * mass / (safe_r * safe_r)
        sin_theta = ti.sin(theta)
        cos_theta = ti.cos(theta)
        safe_sin = sin_theta
        if ti.abs(safe_sin) < 1e-6:
            safe_sin = 1e-6 if cos_theta >= 0.0 else -1e-6

        pt = mom[0]
        pr = mom[1]
        ptheta = mom[2]
        pphi = mom[3]

        pos_dot = mom

        dp_t = -(dlapse_dr / lapse) * pt * pr
        dp_r = (
            -0.5 * lapse * dlapse_dr * pt * pt
            + (dlapse_dr / (2.0 * lapse)) * pr * pr
            + lapse * safe_r * (ptheta * ptheta + sin_theta * sin_theta * pphi * pphi)
        )
        dp_theta = -2.0 * pr * ptheta / safe_r + sin_theta * cos_theta * pphi * pphi
        dp_phi = -(
            2.0 * pr * pphi / safe_r + 2.0 * cos_theta / safe_sin * ptheta * pphi
        )

        mom_dot = vec4f([dp_t, dp_r, dp_theta, dp_phi])
        return pos_dot, mom_dot

    @ti.func
    def _rk4_step(self, pos: vec4f, mom: vec4f) -> tuple[vec4f, vec4f]:
        h = ti.static(self.step_size)

        def blend(base_pos, base_mom, diff_pos, diff_mom, scale):
            return (
                base_pos + diff_pos * scale,
                base_mom + diff_mom * scale,
            )

        k1_p, k1_m = self._geodesic_rhs(pos, mom)
        k2_p, k2_m = self._geodesic_rhs(*blend(pos, mom, k1_p, k1_m, 0.5 * h))
        k3_p, k3_m = self._geodesic_rhs(*blend(pos, mom, k2_p, k2_m, 0.5 * h))
        k4_p, k4_m = self._geodesic_rhs(*blend(pos, mom, k3_p, k3_m, h))

        pos_out = pos + (h / 6.0) * (
            k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p
        )
        mom_out = mom + (h / 6.0) * (
            k1_m + 2.0 * k2_m + 2.0 * k3_m + k4_m
        )

        # keep theta in a valid range to limit coordinate singularities
        pos_out[2] = ti.min(ti.max(pos_out[2], self.theta_min), self.theta_max)
        return pos_out, mom_out

    @ti.func
    def _to_cartesian(self, r: ti.f32, theta: ti.f32, phi: ti.f32) -> vec3f:
        sin_theta = ti.sin(theta)
        return vec3f([
            r * sin_theta * ti.cos(phi),
            r * sin_theta * ti.sin(phi),
            r * ti.cos(theta),
        ])

    @ti.func
    def _rotate_to_disk_frame(self, value: vec3f) -> vec3f:
        s = ti.static(self.disk_sin)
        c = ti.static(self.disk_cos)
        return vec3f([
            value[0],
            value[1] * c + value[2] * s,
            -value[1] * s + value[2] * c,
        ])

    @ti.func
    def _world_dir_to_spherical(
        self,
        r: ti.f32,
        theta: ti.f32,
        phi: ti.f32,
        direction: vec3f,
    ) -> vec3f:
        sin_theta = ti.sin(theta)
        cos_theta = ti.cos(theta)
        sin_phi = ti.sin(phi)
        cos_phi = ti.cos(phi)
        dx = direction[0]
        dy = direction[1]
        dz = direction[2]

        dr = (
            dx * sin_theta * cos_phi
            + dy * sin_theta * sin_phi
            + dz * cos_theta
        )
        denom_r = ti.max(r, 1e-4)
        dtheta = (
            dx * cos_theta * cos_phi
            + dy * cos_theta * sin_phi
            - dz * sin_theta
        ) / denom_r
        denom_phi = ti.max(denom_r * ti.max(ti.abs(sin_theta), 1e-4), 1e-4)
        dphi = (-dx * sin_phi + dy * cos_phi) / denom_phi
        return vec3f([dr, dtheta, dphi])

    @ti.func
    def _disk_intersection(
        self,
        prev_cart: vec3f,
        next_cart: vec3f,
        prev_dist: ti.f32,
        next_dist: ti.f32,
    ) -> tuple[ti.i32, vec3f, ti.f32]:
        hit = 0
        point = vec3f([0.0, 0.0, 0.0])
        radius = 0.0
        cross = (prev_dist > 0.0 and next_dist < 0.0) or (prev_dist < 0.0 and next_dist > 0.0)
        near_plane = ti.abs(next_dist) < self.plane_epsilon
        if cross or near_plane:
            denom = prev_dist - next_dist
            t = 0.0
            if ti.abs(denom) > 1e-6:
                t = prev_dist / denom
            t = ti.min(ti.max(t, 0.0), 1.0)
            point = prev_cart + (next_cart - prev_cart) * t
            local = self._rotate_to_disk_frame(point)
            radius = ti.sqrt(ti.max(local[0] * local[0] + local[1] * local[1], 0.0))
            inner = ti.static(self.disk_inner)
            outer = ti.static(self.disk_outer)
            if (radius >= inner) and (radius <= outer):
                hit = 1
        return hit, point, radius

    @ti.func
    def _shade_disk(self, pos: vec4f, mom: vec4f, radius: ti.f32) -> vec3f:
        metric = self._metric_diagonal(pos[1], pos[2])
        observer = self.observer_four_velocity[None]
        emitter = self._emitter_four_velocity(pos[1])
        denom = self._covariant_dot(metric, emitter, mom)
        denom_sign = 1.0
        if denom < 0.0:
            denom_sign = -1.0
        denom = denom_sign * ti.max(ti.abs(denom), 1e-5)
        g_factor = self._covariant_dot(metric, observer, mom) / denom
        g_factor = ti.min(ti.max(g_factor, 0.05), 5.0)

        inner = ti.static(self.disk_inner)
        outer = ti.static(self.disk_outer)
        span = ti.max(outer - inner, 1e-4)
        t = ti.min(ti.max((radius - inner) / span, 0.0), 1.0)
        color_inner = vec3f([1.25, 0.75, 0.45])
        color_outer = vec3f([0.55, 0.35, 0.25])
        base_color = color_inner * (1.0 - t) + color_outer * t

        intensity = ti.pow(ti.max(1.0 - (radius - inner) / span, 0.0), 1.35)
        doppler = ti.pow(g_factor, 3.0)
        raw = base_color * intensity * doppler

        blueshift = ti.clamp(g_factor - 1.0, -0.6, 1.5)
        tint = vec3f([
            1.0 + 0.65 * blueshift,
            1.0 + 0.25 * blueshift,
            1.0 - 0.9 * blueshift,
        ])
        return raw * tint

    @ti.func
    def _shade_escape(self, pos: vec4f, mom: vec4f) -> vec3f:
        theta = pos[2]
        phi = pos[3]
        two_pi = ti.static(2.0 * math.pi)
        phi = phi - two_pi * ti.floor(phi / two_pi)
        u = phi / two_pi
        v = ti.max(theta / ti.static(math.pi), 0.0)

        density = ti.static(self.grid_density)
        width = ti.static(self.grid_line_width)
        line_u = ti.abs(_fract(u * density) - 0.5)
        line_v = ti.abs(_fract(v * density) - 0.5)
        mask = ti.exp(-ti.min(line_u, line_v) * width)

        radial = pos[1]
        falloff = 1.0 / (1.0 + 0.002 * radial * radial)
        base = vec3f([0.05, 0.08, 0.1]) * falloff
        grid = vec3f([1.0, 0.96, 0.9]) * mask

        # subtle highlight from photon energy compared to static observer
        metric = self._metric_diagonal(pos[1], pos[2])
        observer = self.observer_four_velocity[None]
        energy = self._covariant_dot(metric, observer, mom)
        energy_boost = ti.clamp(ti.abs(energy) * 0.05, 0.0, 0.6)
        highlight = vec3f([0.4, 0.45, 0.5]) * energy_boost

        return base + grid + highlight

    @ti.func
    def _tone_map(self, color: vec3f) -> vec3f:
        exposure = self.exposure[None]
        gamma = self.gamma[None]
        scale = ti.pow(2.0, exposure)
        mapped = 1.0 - ti.exp(-color * scale)
        mapped = ti.max(mapped, 0.0)
        inv_gamma = 1.0 / ti.max(gamma, 1e-3)
        mapped = ti.pow(mapped, inv_gamma)
        return ti.min(mapped, 1.0)

    @ti.func
    def trace_primary(self, px: ti.i32, py: ti.i32) -> vec3f:
        forward = self.forward[None]
        right = self.right[None]
        up = self.up[None]
        origin = self.origin[None]
        spherical = self.camera_spherical[None]

        width_f = ti.static(float(self.width))
        height_f = ti.static(float(self.height))
        ndc_x = ((ti.cast(px, ti.f32) + 0.5) / width_f) * 2.0 - 1.0
        ndc_y = 1.0 - ((ti.cast(py, ti.f32) + 0.5) / height_f) * 2.0

        tan_half_fov = ti.static(self.tan_half_fov)
        aspect = ti.static(self.aspect)
        dir_camera = forward + (
            right * (ndc_x * tan_half_fov * aspect)
        ) + (up * (ndc_y * tan_half_fov))
        direction = self._normalize3(dir_camera)

        r0 = spherical[0]
        theta0 = spherical[1]
        phi0 = spherical[2]

        spatial = self._world_dir_to_spherical(r0, theta0, phi0, direction)
        metric = self._metric_diagonal(r0, theta0)
        spatial_norm = (
            metric[1] * spatial[0] * spatial[0]
            + metric[2] * spatial[1] * spatial[1]
            + metric[3] * spatial[2] * spatial[2]
        )
        pt = ti.sqrt(ti.max(spatial_norm / (-metric[0]), 1e-8))
        mom = vec4f([pt, spatial[0], spatial[1], spatial[2]])
        pos = vec4f([0.0, r0, theta0, phi0])

        prev_cart = origin
        prev_dist = prev_cart.dot(self.disk_normal[None])
        color = vec3f([0.0, 0.0, 0.0])

        for _ in range(ti.static(self.max_steps)):
            if pos[1] <= self.horizon_radius:
                break
            if pos[1] >= self.escape_radius:
                color = self._shade_escape(pos, mom)
                break

            next_pos, next_mom = self._rk4_step(pos, mom)
            next_cart = self._to_cartesian(next_pos[1], next_pos[2], next_pos[3])
            disk_normal = self.disk_normal[None]
            next_dist = next_cart.dot(disk_normal)

            hit, hit_point, radius = self._disk_intersection(
                prev_cart, next_cart, prev_dist, next_dist
            )
            if hit:
                color = self._shade_disk(next_pos, next_mom, radius)
                break

            if next_pos[1] <= self.horizon_radius:
                break
            if next_pos[1] >= self.escape_radius:
                color = self._shade_escape(next_pos, next_mom)
                break

            pos = next_pos
            mom = next_mom
            prev_cart = next_cart
            prev_dist = next_dist

        return self._tone_map(color)

    @ti.kernel
    def render(self):
        for x, y in self.color:
            self.color[x, y] = self.trace_primary(x, y)

    def update_camera(self, dt: float) -> None:  # noqa: ARG002
        frame = build_camera_frame(self.config)
        spherical = _cartesian_to_spherical(frame)
        self.frame = frame
        self._upload_camera_frame(frame, spherical)

    def set_exposure(self, value: float) -> None:
        self.exposure[None] = value

    def adjust_exposure(self, delta: float) -> None:
        self.exposure[None] = self.exposure[None] + delta

    def set_gamma(self, value: float) -> None:
        self.gamma[None] = max(value, 1e-3)

    def readback(self):
        return self.color.to_numpy()
