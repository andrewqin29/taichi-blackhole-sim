"""Physics helpers for bending rays in Schwarzschild spacetime."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

Coord4 = Tuple[float, float, float, float]


@dataclass
class GeodesicState:
    position: Coord4
    momentum: Coord4


def schwarzschild_metric(pos: Coord4, mass: float, horizon_epsilon: float = 1e-8):
    # g_mu_nu at a point; we stick to diagonal form in (t, r, theta, phi)
    _, r, theta, _ = pos
    if r < 2.0 * mass + horizon_epsilon:
        r = 2.0 * mass + horizon_epsilon

    lapse = 1.0 - (2.0 * mass) / r
    sin_theta = math.sin(theta)
    sin_sq = sin_theta * sin_theta

    return [
        [-lapse, 0.0, 0.0, 0.0],
        [0.0, 1.0 / lapse, 0.0, 0.0],
        [0.0, 0.0, r * r, 0.0],
        [0.0, 0.0, 0.0, r * r * sin_sq],
    ]


def inverse_metric(metric):
    # just invert the diagonal elements, sanity-checking zeros
    g_tt = metric[0][0]
    g_rr = metric[1][1]
    g_thth = metric[2][2]
    g_phph = metric[3][3]

    if not all(abs(x) > 1e-12 for x in (g_tt, g_rr, g_thth, g_phph)):
        raise ValueError("metric inversion hit a zero-ish entry")

    return [
        [1.0 / g_tt, 0.0, 0.0, 0.0],
        [0.0, 1.0 / g_rr, 0.0, 0.0],
        [0.0, 0.0, 1.0 / g_thth, 0.0],
        [0.0, 0.0, 0.0, 1.0 / g_phph],
    ]


def christoffel_symbols(pos: Coord4, mass: float, tiny: float = 1e-12):
    # sparse dict: keys are (mu, nu, sigma)
    T, R, TH, PH = 0, 1, 2, 3
    _, r, theta, _ = pos

    lapse = 1.0 - 2.0 * mass / r
    dlapse_dr = 2.0 * mass / (r * r)
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    sin_sq = sin_t * sin_t

    safe_sin = sin_t if abs(sin_t) > tiny else (tiny if cos_t >= 0.0 else -tiny)

    gamma: Dict[Tuple[int, int, int], float] = {}

    def set_sym(mu, nu, sigma, value):
        gamma[(mu, nu, sigma)] = value
        if sigma != nu:
            gamma[(mu, sigma, nu)] = value

    set_sym(T, T, R, dlapse_dr / (2.0 * lapse))
    set_sym(R, T, T, 0.5 * lapse * dlapse_dr)
    set_sym(R, R, R, -dlapse_dr / (2.0 * lapse))
    set_sym(R, TH, TH, -lapse * r)
    set_sym(R, PH, PH, -lapse * r * sin_sq)
    set_sym(TH, R, TH, 1.0 / r)
    set_sym(PH, R, PH, 1.0 / r)
    set_sym(TH, PH, PH, -sin_t * cos_t)
    set_sym(PH, TH, PH, cos_t / safe_sin)

    return gamma


def geodesic_rhs(
    state: GeodesicState,
    mass: float,
    connection_fn: Callable[[Coord4, float], Dict[Tuple[int, int, int], float]],
):
    # derivative wrt affine parameter: dx/dl = p, dp/dl = -Gamma * p * p
    gamma = connection_fn(state.position, mass)
    dp = [0.0, 0.0, 0.0, 0.0]
    for (mu, nu, sigma), value in gamma.items():
        dp[mu] -= value * state.momentum[nu] * state.momentum[sigma]

    return GeodesicState(position=state.momentum, momentum=tuple(dp))


def rk4_step(
    state: GeodesicState,
    mass: float,
    step_size: float,
    connection_fn: Callable[[Coord4, float], Dict[Tuple[int, int, int], float]],
):
    # basic rk4 integrator, expanded inline
    def blend(base, diff, scale):
        return GeodesicState(
            position=tuple(base.position[i] + scale * diff.position[i] for i in range(4)),
            momentum=tuple(base.momentum[i] + scale * diff.momentum[i] for i in range(4)),
        )

    k1 = geodesic_rhs(state, mass, connection_fn)
    k2 = geodesic_rhs(blend(state, k1, 0.5 * step_size), mass, connection_fn)
    k3 = geodesic_rhs(blend(state, k2, 0.5 * step_size), mass, connection_fn)
    k4 = geodesic_rhs(blend(state, k3, step_size), mass, connection_fn)

    pos = []
    mom = []
    for i in range(4):
        pos.append(
            state.position[i]
            + (step_size / 6.0)
            * (
                k1.position[i]
                + 2.0 * k2.position[i]
                + 2.0 * k3.position[i]
                + k4.position[i]
            )
        )
        mom.append(
            state.momentum[i]
            + (step_size / 6.0)
            * (
                k1.momentum[i]
                + 2.0 * k2.momentum[i]
                + 2.0 * k3.momentum[i]
                + k4.momentum[i]
            )
        )

    return GeodesicState(position=tuple(pos), momentum=tuple(mom))


def integrate_geodesic(
    initial_state: GeodesicState,
    mass: float,
    step_size: float,
    max_steps: int,
    connection_fn: Callable[[Coord4, float], Dict[Tuple[int, int, int], float]],
    termination_fn: Callable[[GeodesicState], bool],
):
    # march the ray until something interesting happens
    state = initial_state
    for step_idx in range(max_steps):
        if termination_fn(state):
            return state, step_idx
        state = rk4_step(state, mass, step_size, connection_fn)
    return state, max_steps


def redshift_factor(
    observer_four_velocity: Coord4,
    emitter_four_velocity: Coord4,
    photon_momentum: Coord4,
    metric,
):
    # g = (u_obs dot p) / (u_em dot p)
    def covariant_dot(a, b):
        total = 0.0
        for mu in range(4):
            for nu in range(4):
                total += metric[mu][nu] * a[mu] * b[nu]
        return total

    num = covariant_dot(observer_four_velocity, photon_momentum)
    denom = covariant_dot(emitter_four_velocity, photon_momentum)
    if abs(denom) < 1e-12:
        raise ZeroDivisionError("redshift factor blew up; emitter dot photon ~= 0")
    return num / denom

