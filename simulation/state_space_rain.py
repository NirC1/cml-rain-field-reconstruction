"""Minimal state-space rain simulation.

This module implements the smallest useful version of:

    x[k + 1] = F x[k]

where x is a flattened 2D rain field. The transition matrix F is built from a
first-order upwind advection term, a diffusion term, and optional decay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StateGrid:
    """Cell-centered 2D grid for a rain state vector."""

    nx: int
    ny: int
    dx_km: float
    dy_km: float

    @property
    def size(self) -> int:
        return self.nx * self.ny

    @property
    def x_km(self) -> np.ndarray:
        return (np.arange(self.nx) + 0.5) * self.dx_km

    @property
    def y_km(self) -> np.ndarray:
        return (np.arange(self.ny) + 0.5) * self.dy_km

    def flatten(self, field: np.ndarray) -> np.ndarray:
        return np.asarray(field, dtype=float).reshape(self.size)

    def unflatten(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=float).reshape(self.ny, self.nx)


def gaussian_initial_field(
    grid: StateGrid,
    *,
    center_x_km: float,
    center_y_km: float,
    sigma_x_km: float,
    sigma_y_km: float,
    intensity_mm_h: float,
) -> np.ndarray:
    """Create a 2D Gaussian rain-rate field in mm/hour."""

    xx, yy = np.meshgrid(grid.x_km, grid.y_km)
    exponent = (
        ((xx - center_x_km) ** 2) / (2.0 * sigma_x_km**2)
        + ((yy - center_y_km) ** 2) / (2.0 * sigma_y_km**2)
    )
    return intensity_mm_h * np.exp(-exponent)


def build_transition_matrix(
    grid: StateGrid,
    *,
    dt_h: float,
    wind_velocity_km_h: tuple[float, float],
    diffusion_km2_h: float = 0.0,
    decay_h_inv: float = 0.0,
) -> np.ndarray:
    """Build dense F for x[k + 1] = F x[k].

    Boundary condition: periodic. This is intentionally simple for early
    testing and keeps transported rain mass inside the simulation domain.
    """

    u_km_h, v_km_h = wind_velocity_km_h
    cfl_x = abs(u_km_h) * dt_h / grid.dx_km
    cfl_y = abs(v_km_h) * dt_h / grid.dy_km
    diff_x = diffusion_km2_h * dt_h / grid.dx_km**2
    diff_y = diffusion_km2_h * dt_h / grid.dy_km**2
    decay = decay_h_inv * dt_h

    center_weight = 1.0 - cfl_x - cfl_y - 2.0 * diff_x - 2.0 * diff_y - decay
    if center_weight < 0.0:
        raise ValueError(
            "Unstable explicit transition: reduce dt_h, wind speed, or diffusion."
        )

    transition = np.zeros((grid.size, grid.size), dtype=float)

    def index(ix: int, iy: int) -> int:
        return (iy % grid.ny) * grid.nx + (ix % grid.nx)

    for iy in range(grid.ny):
        for ix in range(grid.nx):
            row = index(ix, iy)
            transition[row, index(ix, iy)] += center_weight

            if u_km_h >= 0.0:
                transition[row, index(ix - 1, iy)] += cfl_x
            else:
                transition[row, index(ix + 1, iy)] += cfl_x

            if v_km_h >= 0.0:
                transition[row, index(ix, iy - 1)] += cfl_y
            else:
                transition[row, index(ix, iy + 1)] += cfl_y

            transition[row, index(ix - 1, iy)] += diff_x
            transition[row, index(ix + 1, iy)] += diff_x
            transition[row, index(ix, iy - 1)] += diff_y
            transition[row, index(ix, iy + 1)] += diff_y

    return transition


def simulate_state_space_rain(
    *,
    grid: StateGrid,
    initial_field_mm_h: np.ndarray,
    transition_matrix: np.ndarray,
    num_steps: int,
) -> np.ndarray:
    """Simulate rain dynamics and return R[t, y, x] in mm/hour."""

    state = grid.flatten(initial_field_mm_h)
    values = np.zeros((num_steps + 1, grid.ny, grid.nx), dtype=float)
    values[0] = grid.unflatten(state)

    for step in range(1, num_steps + 1):
        state = transition_matrix @ state
        state = np.maximum(state, 0.0)
        values[step] = grid.unflatten(state)

    return values

