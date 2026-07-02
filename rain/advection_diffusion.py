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

    mu_km: np.ndarray
    covariance_km2: np.ndarray
    intensity_mm_h: float

    def __post_init__(self) -> None:
        mu_km = np.asarray(self.mu_km, dtype=float)
        covariance_km2 = np.asarray(self.covariance_km2, dtype=float)

        if mu_km.shape != (2,):
            raise ValueError("mu_km must be a numpy array with shape (2,).")
        if covariance_km2.shape != (2, 2):
            raise ValueError(
                "covariance_km2 must be a numpy array with shape (2, 2)."
            )
        if not np.allclose(covariance_km2, covariance_km2.T):
            raise ValueError("covariance_km2 must be symmetric.")
        if np.any(np.linalg.eigvalsh(covariance_km2) <= 0.0):
            raise ValueError("covariance_km2 must be positive definite.")
        if self.intensity_mm_h < 0.0:
            raise ValueError("intensity_mm_h must be nonnegative.")

        mu_km.setflags(write=False)
        covariance_km2.setflags(write=False)
        object.__setattr__(self, "mu_km", mu_km)
        object.__setattr__(self, "covariance_km2", covariance_km2)


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
    wind_field_km_h: np.ndarray | None = None
    diffusion_km2_h: float = 0.04
    decay_h_inv: float = 0.02
    process_noise_std_mm_h_sqrt_h: float = 0.0
    process_noise_sigma_km: float = 1.0
    boundary_condition: BoundaryCondition = "periodic"
    initial_conditions: tuple[GaussianInitialCondition, ...] = (
        GaussianInitialCondition(
            mu_km=np.array([8.0, 15.0]),
            covariance_km2=np.array([[1.4**2, 0.0], 
                                     [0.0, 2.0**2]]),
            intensity_mm_h=35.0,
        ),
        GaussianInitialCondition(
            mu_km=np.array([18.0, 9.0]),
            covariance_km2=np.array([[2.0**2, 0.0], 
                                     [0.0, 1.2**2]]),
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
    rng = np.random.default_rng(config.random_seed)

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
            wind_field_km_h=config.wind_field_km_h,
            diffusion_km2_h=config.diffusion_km2_h,
            decay_h_inv=config.decay_h_inv,
            process_noise_std_mm_h_sqrt_h=config.process_noise_std_mm_h_sqrt_h,
            process_noise_sigma_km=config.process_noise_sigma_km,
            rng=rng,
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
            "wind_field_km_h_shape": (
                None
                if config.wind_field_km_h is None
                else np.asarray(config.wind_field_km_h).shape
            ),
            "diffusion_km2_h": config.diffusion_km2_h,
            "decay_h_inv": config.decay_h_inv,
            "process_noise_std_mm_h_sqrt_h": config.process_noise_std_mm_h_sqrt_h,
            "process_noise_sigma_km": config.process_noise_sigma_km,
            "boundary_condition": config.boundary_condition,
            "initial_conditions": [
                _initial_condition_metadata(condition)
                for condition in config.initial_conditions
            ],
            "update": "explicit_upwind_advection_centered_diffusion",
        },
        metadata={
            "array_convention": "values_mm_h[t, y, x]",
            "grid_convention": "cell_centered_pixels",
            "pde": "dR/dt = -div(u R) + D laplacian(R) - lambda R + eta",
            "state_space_view": "x[k+1] = F x[k] + w[k], applied matrix-free",
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
        dx = xx - condition.mu_km[0]
        dy = yy - condition.mu_km[1]
        offsets = np.stack((dx, dy), axis=-1)
        covariance_inv = np.linalg.inv(condition.covariance_km2)
        exponent = 0.5 * np.einsum(
            "...i,ij,...j->...",
            offsets,
            covariance_inv,
            offsets,
        )
        field += condition.intensity_mm_h * np.exp(-exponent)

    return field


def _initial_condition_metadata(
    condition: GaussianInitialCondition,
) -> dict[str, float | list[float] | list[list[float]]]:
    return {
        "mu_km": condition.mu_km.tolist(),
        "covariance_km2": condition.covariance_km2.tolist(),
        "intensity_mm_h": condition.intensity_mm_h,
    }


def advection_diffusion_step(
    field_mm_h: np.ndarray,
    *,
    grid: GridSpec,
    dt_h: float,
    wind_velocity_km_h: tuple[float, float],
    diffusion_km2_h: float,
    decay_h_inv: float,
    wind_field_km_h: np.ndarray | None = None,
    process_noise_std_mm_h_sqrt_h: float = 0.0,
    process_noise_sigma_km: float = 1.0,
    rng: np.random.Generator | None = None,
    boundary_condition: BoundaryCondition = "periodic",
) -> np.ndarray:
    """Advance one explicit finite-difference step."""

    if boundary_condition != "periodic":
        raise NotImplementedError("Only periodic boundaries are implemented for now.")

    if wind_field_km_h is None:
        u_km_h, v_km_h = wind_velocity_km_h
        d_rain_dx = _upwind_x_derivative(field_mm_h, grid.dx_km, u_km_h)
        d_rain_dy = _upwind_y_derivative(field_mm_h, grid.dy_km, v_km_h)
        advection = -u_km_h * d_rain_dx - v_km_h * d_rain_dy
    else:
        _validate_wind_field(wind_field_km_h, grid)
        advection = -_upwind_flux_divergence(
            field_mm_h,
            wind_field_km_h,
            grid=grid,
        )
    laplacian = _periodic_laplacian(field_mm_h, grid.dx_km, grid.dy_km)

    tendency = (
        advection
        + diffusion_km2_h * laplacian
        - decay_h_inv * field_mm_h
    )
    next_field = field_mm_h + dt_h * tendency
    if process_noise_std_mm_h_sqrt_h > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        next_field = next_field + _smoothed_process_noise(
            shape=field_mm_h.shape,
            grid=grid,
            std_mm_h_sqrt_h=process_noise_std_mm_h_sqrt_h,
            dt_h=dt_h,
            sigma_km=process_noise_sigma_km,
            rng=rng,
        )
    return next_field


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


def _upwind_flux_divergence(
    field: np.ndarray,
    wind_field_km_h: np.ndarray,
    *,
    grid: GridSpec,
) -> np.ndarray:
    """Return div(u R) using first-order upwind fluxes at cell faces."""

    wind_field = np.asarray(wind_field_km_h, dtype=float)
    u_cell = wind_field[..., 0]
    v_cell = wind_field[..., 1]

    u_right = 0.5 * (u_cell + np.roll(u_cell, shift=-1, axis=1))
    r_right = np.where(u_right >= 0.0, field, np.roll(field, shift=-1, axis=1))
    flux_x_right = u_right * r_right
    flux_x_left = np.roll(flux_x_right, shift=1, axis=1)

    v_top = 0.5 * (v_cell + np.roll(v_cell, shift=-1, axis=0))
    r_top = np.where(v_top >= 0.0, field, np.roll(field, shift=-1, axis=0))
    flux_y_top = v_top * r_top
    flux_y_bottom = np.roll(flux_y_top, shift=1, axis=0)

    return (
        (flux_x_right - flux_x_left) / grid.dx_km
        + (flux_y_top - flux_y_bottom) / grid.dy_km
    )


def _smoothed_process_noise(
    *,
    shape: tuple[int, int],
    grid: GridSpec,
    std_mm_h_sqrt_h: float,
    dt_h: float,
    sigma_km: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate Gaussian-smoothed additive model-error noise."""

    noise = rng.normal(
        loc=0.0,
        scale=std_mm_h_sqrt_h * np.sqrt(dt_h),
        size=shape,
    )
    return _gaussian_convolve_periodic(
        noise,
        sigma_x_cells=sigma_km / grid.dx_km,
        sigma_y_cells=sigma_km / grid.dy_km,
    )


def _gaussian_convolve_periodic(
    field: np.ndarray,
    *,
    sigma_x_cells: float,
    sigma_y_cells: float,
) -> np.ndarray:
    """Convolve a 2D field with a separable Gaussian using periodic boundaries."""

    if sigma_x_cells == 0.0 and sigma_y_cells == 0.0:
        return field

    result = field
    if sigma_x_cells > 0.0:
        kernel_x = _gaussian_kernel_1d(sigma_x_cells)
        result = _convolve_periodic_1d(result, kernel_x, axis=1)
    if sigma_y_cells > 0.0:
        kernel_y = _gaussian_kernel_1d(sigma_y_cells)
        result = _convolve_periodic_1d(result, kernel_y, axis=0)
    return result


def _gaussian_kernel_1d(sigma_cells: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma_cells)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma_cells**2))
    return kernel / kernel.sum()


def _convolve_periodic_1d(
    field: np.ndarray,
    kernel: np.ndarray,
    *,
    axis: int,
) -> np.ndarray:
    radius = len(kernel) // 2
    result = np.zeros_like(field, dtype=np.float64)
    for offset, weight in zip(range(-radius, radius + 1), kernel):
        result += weight * np.roll(field, shift=offset, axis=axis)
    return result


def _validate_config(config: AdvectionDiffusionConfig) -> None:
    if config.diffusion_km2_h < 0.0:
        raise ValueError("diffusion_km2_h must be nonnegative.")
    if config.decay_h_inv < 0.0:
        raise ValueError("decay_h_inv must be nonnegative.")
    if config.process_noise_std_mm_h_sqrt_h < 0.0:
        raise ValueError("process_noise_std_mm_h_sqrt_h must be nonnegative.")
    if config.process_noise_sigma_km < 0.0:
        raise ValueError("process_noise_sigma_km must be nonnegative.")
    if config.wind_field_km_h is not None:
        _validate_wind_field(config.wind_field_km_h, config.grid)

    max_abs_u_km_h, max_abs_v_km_h = _wind_speed_limits(config)
    cfl_x = max_abs_u_km_h * config.time.dt_h / config.grid.dx_km
    cfl_y = max_abs_v_km_h * config.time.dt_h / config.grid.dy_km
    diff_x = config.diffusion_km2_h * config.time.dt_h / config.grid.dx_km**2
    diff_y = config.diffusion_km2_h * config.time.dt_h / config.grid.dy_km**2
    stability_margin = 1.0 - cfl_x - cfl_y - 2.0 * diff_x - 2.0 * diff_y

    if stability_margin < 0.0:
        raise ValueError(
            "Unstable explicit advection-diffusion config: reduce dt_h, "
            "wind speed, or diffusion."
        )


def _validate_wind_field(wind_field_km_h: np.ndarray, grid: GridSpec) -> None:
    wind_field = np.asarray(wind_field_km_h)
    expected_shape = (grid.ny, grid.nx, 2)
    if wind_field.shape != expected_shape:
        raise ValueError(
            f"wind_field_km_h has shape {wind_field.shape}, "
            f"expected {expected_shape}."
        )
    if not np.all(np.isfinite(wind_field)):
        raise ValueError("wind_field_km_h must contain only finite values.")


def _wind_speed_limits(config: AdvectionDiffusionConfig) -> tuple[float, float]:
    if config.wind_field_km_h is None:
        u_km_h, v_km_h = config.wind_velocity_km_h
        return abs(u_km_h), abs(v_km_h)

    wind_field = np.asarray(config.wind_field_km_h)
    return (
        float(np.max(np.abs(wind_field[..., 0]))),
        float(np.max(np.abs(wind_field[..., 1]))),
    )
