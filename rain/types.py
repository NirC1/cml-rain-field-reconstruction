"""Shared rain-field data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """Physical metadata for a cell-centered rectangular grid."""

    domain_width_km: float
    domain_height_km: float
    dx_km: float
    dy_km: float
    coordinate_system: str = "local_cartesian_km"
    origin_lat: float | None = None
    origin_lon: float | None = None

    @property
    def nx(self) -> int:
        return int(round(self.domain_width_km / self.dx_km))

    @property
    def ny(self) -> int:
        return int(round(self.domain_height_km / self.dy_km))

    @property
    def x_centers_km(self) -> np.ndarray:
        return (np.arange(self.nx) + 0.5) * self.dx_km

    @property
    def y_centers_km(self) -> np.ndarray:
        return (np.arange(self.ny) + 0.5) * self.dy_km

    @property
    def x_edges_km(self) -> np.ndarray:
        return np.arange(self.nx + 1) * self.dx_km

    @property
    def y_edges_km(self) -> np.ndarray:
        return np.arange(self.ny + 1) * self.dy_km

    @property
    def shape_yx(self) -> tuple[int, int]:
        return self.ny, self.nx

    def validate(self) -> None:
        if self.domain_width_km <= 0.0 or self.domain_height_km <= 0.0:
            raise ValueError("Domain dimensions must be positive.")
        if self.dx_km <= 0.0 or self.dy_km <= 0.0:
            raise ValueError("Grid spacing must be positive.")
        if not np.isclose(self.nx * self.dx_km, self.domain_width_km):
            raise ValueError("domain_width_km must be divisible by dx_km.")
        if not np.isclose(self.ny * self.dy_km, self.domain_height_km):
            raise ValueError("domain_height_km must be divisible by dy_km.")


@dataclass(frozen=True)
class TimeSpec:
    """Time metadata for a simulation."""

    duration_h: float
    dt_h: float
    start_time: str | None = None

    @property
    def num_steps(self) -> int:
        return int(round(self.duration_h / self.dt_h)) + 1

    @property
    def t_h(self) -> np.ndarray:
        return np.arange(self.num_steps) * self.dt_h

    def validate(self) -> None:
        if self.duration_h <= 0.0:
            raise ValueError("duration_h must be positive.")
        if self.dt_h <= 0.0:
            raise ValueError("dt_h must be positive.")
        if not np.isclose((self.num_steps - 1) * self.dt_h, self.duration_h):
            raise ValueError("duration_h must be divisible by dt_h.")


@dataclass(frozen=True)
class RainField:
    """Rain-rate field and metadata passed between project modules."""

    values_mm_h: np.ndarray
    grid: GridSpec
    time: TimeSpec
    model_name: str
    units: str = "mm/hour"
    random_seed: int | None = None
    model_params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def x_km(self) -> np.ndarray:
        return self.grid.x_centers_km

    @property
    def y_km(self) -> np.ndarray:
        return self.grid.y_centers_km

    @property
    def t_h(self) -> np.ndarray:
        return self.time.t_h

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.values_mm_h.shape

    def validate(self) -> None:
        self.grid.validate()
        self.time.validate()
        expected = (self.time.num_steps, self.grid.ny, self.grid.nx)
        if self.values_mm_h.shape != expected:
            raise ValueError(
                f"Rain values have shape {self.values_mm_h.shape}, expected {expected}."
            )
        if self.units != "mm/hour":
            raise ValueError("RainField units must currently be 'mm/hour'.")

    def at_grid(self, time_index: int) -> np.ndarray:
        """Return the 2D rain-rate matrix at a discrete time index."""

        return self.values_mm_h[time_index]

    def sample_nearest(self, x_km: float, y_km: float, t_h: float) -> float:
        """Sample R(x, y, t) using nearest-neighbor lookup on the grid."""

        ix = int(np.abs(self.x_km - x_km).argmin())
        iy = int(np.abs(self.y_km - y_km).argmin())
        it = int(np.abs(self.t_h - t_h).argmin())
        return float(self.values_mm_h[it, iy, ix])

    def summary(self) -> dict[str, Any]:
        """Return a compact metadata summary for notebooks and logs."""

        return {
            "shape_TYX": self.shape,
            "domain_km": (
                self.grid.domain_width_km,
                self.grid.domain_height_km,
            ),
            "spacing_km": (self.grid.dx_km, self.grid.dy_km),
            "duration_h": self.time.duration_h,
            "dt_h": self.time.dt_h,
            "units": self.units,
            "model_name": self.model_name,
            "random_seed": self.random_seed,
            "max_mm_h": float(self.values_mm_h.max()),
            "mean_mm_h": float(self.values_mm_h.mean()),
        }

