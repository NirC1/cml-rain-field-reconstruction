"""Rain-field simulation interfaces and implementations."""

from rain.advection_diffusion import (
    AdvectionDiffusionConfig,
    GaussianInitialCondition,
    simulate_advection_diffusion_rain,
)
from rain.types import GridSpec, RainField, TimeSpec

__all__ = [
    "AdvectionDiffusionConfig",
    "GaussianInitialCondition",
    "GridSpec",
    "RainField",
    "TimeSpec",
    "simulate_advection_diffusion_rain",
]

