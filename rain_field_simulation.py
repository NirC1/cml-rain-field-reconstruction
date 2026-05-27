"""Minimal rain-field simulation with moving Gaussian rain cells.

The simulated field represents rain rate R(x, y, t) in mm/hour over a
rectangular domain measured in kilometers. The returned array is discretized as
R[t, y, x], which is the common practical representation for reconstruction and
machine-learning experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianRainCell:
    """A single moving Gaussian rain cell.

    Attributes:
        x0_km: Initial x center in km.
        y0_km: Initial y center in km.
        intensity_mm_h: Peak rain rate in mm/hour at t=0.
        sigma_x_km: Horizontal Gaussian width in km.
        sigma_y_km: Vertical Gaussian width in km.
        velocity_x_km_h: Cell velocity in the x direction, km/hour.
        velocity_y_km_h: Cell velocity in the y direction, km/hour.
        decay_h_inv: Exponential intensity decay rate, 1/hour.
    """

    x0_km: float
    y0_km: float
    intensity_mm_h: float
    sigma_x_km: float
    sigma_y_km: float
    velocity_x_km_h: float = 0.0
    velocity_y_km_h: float = 0.0
    decay_h_inv: float = 0.0


@dataclass(frozen=True)
class RainField:
    """A discretized rain field and its physical grid coordinates."""

    values_mm_h: np.ndarray
    x_km: np.ndarray
    y_km: np.ndarray
    t_h: np.ndarray
    cells: tuple[GaussianRainCell, ...]

    def at_grid(self, time_index: int) -> np.ndarray:
        """Return the 2D rain-rate matrix at a discrete time index."""

        return self.values_mm_h[time_index]

    def sample_nearest(self, x_km: float, y_km: float, t_h: float) -> float:
        """Sample R(x, y, t) using nearest-neighbor lookup on the grid."""

        ix = int(np.abs(self.x_km - x_km).argmin())
        iy = int(np.abs(self.y_km - y_km).argmin())
        it = int(np.abs(self.t_h - t_h).argmin())
        return float(self.values_mm_h[it, iy, ix])


def make_random_gaussian_cells(
    *,
    num_cells: int,
    domain_width_km: float,
    domain_height_km: float,
    intensity_range_mm_h: tuple[float, float] = (5.0, 40.0),
    sigma_range_km: tuple[float, float] = (0.8, 3.0),
    velocity_x_km_h: float = 0.0,
    velocity_y_km_h: float = 0.0,
    decay_h_inv: float = 0.0,
    seed: int | None = None,
) -> tuple[GaussianRainCell, ...]:
    """Create random Gaussian rain cells inside the simulation domain."""

    rng = np.random.default_rng(seed)
    intensities = rng.uniform(*intensity_range_mm_h, size=num_cells)
    sigmas_x = rng.uniform(*sigma_range_km, size=num_cells)
    sigmas_y = rng.uniform(*sigma_range_km, size=num_cells)
    x0 = rng.uniform(0.0, domain_width_km, size=num_cells)
    y0 = rng.uniform(0.0, domain_height_km, size=num_cells)

    return tuple(
        GaussianRainCell(
            x0_km=float(x0[i]),
            y0_km=float(y0[i]),
            intensity_mm_h=float(intensities[i]),
            sigma_x_km=float(sigmas_x[i]),
            sigma_y_km=float(sigmas_y[i]),
            velocity_x_km_h=velocity_x_km_h,
            velocity_y_km_h=velocity_y_km_h,
            decay_h_inv=decay_h_inv,
        )
        for i in range(num_cells)
    )


def simulate_gaussian_rain_field(
    *,
    domain_width_km: float = 20.0,
    domain_height_km: float = 20.0,
    dx_km: float = 0.25,
    dy_km: float = 0.25,
    duration_h: float = 1.0,
    dt_h: float = 1.0 / 60.0,
    cells: tuple[GaussianRainCell, ...] | None = None,
    num_random_cells: int = 4,
    wind_velocity_km_h: tuple[float, float] = (10.0, 0.0),
    decay_h_inv: float = 0.05,
    seed: int | None = None,
) -> RainField:
    """Simulate R(x, y, t) as a sum of moving Gaussian rain cells.

    Returns:
        RainField where ``values_mm_h`` has shape ``[T, Ny, Nx]`` and stores
        rain rate in mm/hour.
    """

    x_km = np.arange(0.0, domain_width_km + dx_km, dx_km)
    y_km = np.arange(0.0, domain_height_km + dy_km, dy_km)
    t_h = np.arange(0.0, duration_h + dt_h, dt_h)

    if cells is None:
        cells = make_random_gaussian_cells(
            num_cells=num_random_cells,
            domain_width_km=domain_width_km,
            domain_height_km=domain_height_km,
            velocity_x_km_h=wind_velocity_km_h[0],
            velocity_y_km_h=wind_velocity_km_h[1],
            decay_h_inv=decay_h_inv,
            seed=seed,
        )

    xx, yy = np.meshgrid(x_km, y_km)
    values = np.zeros((len(t_h), len(y_km), len(x_km)), dtype=np.float64)

    for it, time_h in enumerate(t_h):
        frame = np.zeros_like(xx, dtype=np.float64)

        for cell in cells:
            center_x = cell.x0_km + cell.velocity_x_km_h * time_h
            center_y = cell.y0_km + cell.velocity_y_km_h * time_h
            intensity = cell.intensity_mm_h * np.exp(-cell.decay_h_inv * time_h)

            exponent = (
                ((xx - center_x) ** 2) / (2.0 * cell.sigma_x_km**2)
                + ((yy - center_y) ** 2) / (2.0 * cell.sigma_y_km**2)
            )
            frame += intensity * np.exp(-exponent)

        values[it] = frame

    return RainField(
        values_mm_h=values,
        x_km=x_km,
        y_km=y_km,
        t_h=t_h,
        cells=tuple(cells),
    )


if __name__ == "__main__":
    rain = simulate_gaussian_rain_field(seed=42)
    print("R shape [T, Y, X]:", rain.values_mm_h.shape)
    print("Max rain rate [mm/h]:", float(rain.values_mm_h.max()))
    print("Example R(10 km, 10 km, 0.5 h):", rain.sample_nearest(10.0, 10.0, 0.5))
