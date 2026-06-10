"""Matrix-free advection-diffusion rain-field simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from rain.types import GridSpec, RainField, TimeSpec

BoundaryCondition = Literal["periodic"]


@dataclass(frozen=True)
class GaussianInitialCondition:
    """A Gaussian initial rain field used to seed the PDE simulation."""

    center_x_km: float
    center_y_km: float
    sigma_x_km: float
    sigma_y_km: float
    intensity_mm_h: float


@dataclass(frozen=True)
class AdvectionDiffusionConfig:
    """Configuration for the advection-diffusion rain simulator."""

    grid: GridSpec = field(
        default_factory=lambda: GridSpec(
            domain_width_km=30.0,
            domain_height_km=30.0,
            dx_km=0.5,
            dy_km=0.5,
        )
    )
    time: TimeSpec = field(
        default_factory=lambda: TimeSpec(duration_h=1.0, dt_h=1.0 / 60.0)
    )
    wind_velocity_km_h: tuple[float, float] = (8.0, 3.0)
    diffusion_km2_h: float = 0.04
    decay_h_inv: float = 0.02
    boundary_condition: BoundaryCondition = "periodic"
    initial_conditions: tuple[GaussianInitialCondition, ...] = (
        GaussianInitialCondition(
            center_x_km=8.0,
            center_y_km=15.0,
            sigma_x_km=1.4,
            sigma_y_km=2.0,
            intensity_mm_h=35.0,
        ),
        GaussianInitialCondition(
            center_x_km=18.0,
            center_y_km=9.0,
            sigma_x_km=2.0,
            sigma_y_km=1.2,
            intensity_mm_h=22.0,
        ),
    )
    random_seed: int | None = None


def simulate_advection_diffusion_rain(
    config: AdvectionDiffusionConfig,
    *,
    initial_field_mm_h: np.ndarray | None = None,
) -> RainField:
    """Simulate R(x, y, t) with advection, diffusion, and decay."""

    config.grid.validate()
    config.time.validate()
    _validate_config(config)

    if initial_field_mm_h is None:
        current = build_initial_field(config.grid, config.initial_conditions)
    else:
        current = np.asarray(initial_field_mm_h, dtype=float)
        if current.shape != config.grid.shape_yx:
            raise ValueError(
                f"initial_field_mm_h has shape {current.shape}, "
                f"expected {config.grid.shape_yx}."
            )

    values = np.zeros(
        (config.time.num_steps, config.grid.ny, config.grid.nx),
        dtype=np.float64,
    )
    values[0] = np.maximum(current, 0.0)

    for time_index in range(1, config.time.num_steps):
        current = advection_diffusion_step(
            values[time_index - 1],
            grid=config.grid,
            dt_h=config.time.dt_h,
            wind_velocity_km_h=config.wind_velocity_km_h,
            diffusion_km2_h=config.diffusion_km2_h,
            decay_h_inv=config.decay_h_inv,
            boundary_condition=config.boundary_condition,
        )
        values[time_index] = np.maximum(current, 0.0)

    rain = RainField(
        values_mm_h=values,
        grid=config.grid,
        time=config.time,
        model_name="advection_diffusion",
        random_seed=config.random_seed,
        model_params={
            "wind_velocity_km_h": config.wind_velocity_km_h,
            "diffusion_km2_h": config.diffusion_km2_h,
            "decay_h_inv": config.decay_h_inv,
            "boundary_condition": config.boundary_condition,
            "initial_conditions": [
                condition.__dict__ for condition in config.initial_conditions
            ],
            "update": "explicit_upwind_advection_centered_diffusion",
        },
        metadata={
            "array_convention": "values_mm_h[t, y, x]",
            "grid_convention": "cell_centered_pixels",
            "pde": "dR/dt = -u R_x - v R_y + D laplacian(R) - lambda R",
            "state_space_view": "x[k+1] = F x[k], applied matrix-free",
            "intended_downstream_modules": [
                "sensor_map_generator",
                "measurement_simulator",
                "reconstruction_model",
                "evaluator",
            ],
        },
    )
    rain.validate()
    return rain


def build_initial_field(
    grid: GridSpec,
    initial_conditions: tuple[GaussianInitialCondition, ...],
) -> np.ndarray:
    """Build an initial rain field from one or more Gaussian rain cells."""

    xx, yy = np.meshgrid(grid.x_centers_km, grid.y_centers_km)
    field = np.zeros(grid.shape_yx, dtype=np.float64)

    for condition in initial_conditions:
        exponent = (
            ((xx - condition.center_x_km) ** 2) / (2.0 * condition.sigma_x_km**2)
            + ((yy - condition.center_y_km) ** 2) / (2.0 * condition.sigma_y_km**2)
        )
        field += condition.intensity_mm_h * np.exp(-exponent)

    return field


def advection_diffusion_step(
    field_mm_h: np.ndarray,
    *,
    grid: GridSpec,
    dt_h: float,
    wind_velocity_km_h: tuple[float, float],
    diffusion_km2_h: float,
    decay_h_inv: float,
    boundary_condition: BoundaryCondition = "periodic",
) -> np.ndarray:
    """Advance one explicit finite-difference step."""

    if boundary_condition != "periodic":
        raise NotImplementedError("Only periodic boundaries are implemented for now.")

    u_km_h, v_km_h = wind_velocity_km_h
    d_rain_dx = _upwind_x_derivative(field_mm_h, grid.dx_km, u_km_h)
    d_rain_dy = _upwind_y_derivative(field_mm_h, grid.dy_km, v_km_h)
    laplacian = _periodic_laplacian(field_mm_h, grid.dx_km, grid.dy_km)

    tendency = (
        -u_km_h * d_rain_dx
        - v_km_h * d_rain_dy
        + diffusion_km2_h * laplacian
        - decay_h_inv * field_mm_h
    )
    return field_mm_h + dt_h * tendency


def _upwind_x_derivative(field: np.ndarray, dx_km: float, u_km_h: float) -> np.ndarray:
    if u_km_h >= 0.0:
        return (field - np.roll(field, shift=1, axis=1)) / dx_km
    return (np.roll(field, shift=-1, axis=1) - field) / dx_km


def _upwind_y_derivative(field: np.ndarray, dy_km: float, v_km_h: float) -> np.ndarray:
    if v_km_h >= 0.0:
        return (field - np.roll(field, shift=1, axis=0)) / dy_km
    return (np.roll(field, shift=-1, axis=0) - field) / dy_km


def _periodic_laplacian(field: np.ndarray, dx_km: float, dy_km: float) -> np.ndarray:
    second_x = (
        np.roll(field, shift=-1, axis=1)
        - 2.0 * field
        + np.roll(field, shift=1, axis=1)
    ) / dx_km**2
    second_y = (
        np.roll(field, shift=-1, axis=0)
        - 2.0 * field
        + np.roll(field, shift=1, axis=0)
    ) / dy_km**2
    return second_x + second_y


def _validate_config(config: AdvectionDiffusionConfig) -> None:
    if config.diffusion_km2_h < 0.0:
        raise ValueError("diffusion_km2_h must be nonnegative.")
    if config.decay_h_inv < 0.0:
        raise ValueError("decay_h_inv must be nonnegative.")

    u_km_h, v_km_h = config.wind_velocity_km_h
    cfl_x = abs(u_km_h) * config.time.dt_h / config.grid.dx_km
    cfl_y = abs(v_km_h) * config.time.dt_h / config.grid.dy_km
    diff_x = config.diffusion_km2_h * config.time.dt_h / config.grid.dx_km**2
    diff_y = config.diffusion_km2_h * config.time.dt_h / config.grid.dy_km**2
    stability_margin = 1.0 - cfl_x - cfl_y - 2.0 * diff_x - 2.0 * diff_y

    if stability_margin < 0.0:
        raise ValueError(
            "Unstable explicit advection-diffusion config: reduce dt_h, "
            "wind speed, or diffusion."
        )

