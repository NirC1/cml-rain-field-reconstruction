"""Compatibility facade for the rain simulator.

New code should import from ``rain`` directly. This file remains so older
notebooks and scripts keep working while the project structure settles.
"""

from __future__ import annotations

import numpy as np

from rain import (
    AdvectionDiffusionConfig,
    GaussianInitialCondition,
    GridSpec,
    RainField,
    TimeSpec,
    simulate_advection_diffusion_rain,
)


def simulate_rain_field(
    *,
    domain_width_km: float = 30.0,
    domain_height_km: float = 30.0,
    dx_km: float = 0.5,
    dy_km: float = 0.5,
    duration_h: float = 1.0,
    dt_h: float = 1.0 / 60.0,
    wind_velocity_km_h: tuple[float, float] = (8.0, 3.0),
    wind_field_km_h: np.ndarray | None = None,
    diffusion_km2_h: float = 0.04,
    decay_h_inv: float = 0.02,
    process_noise_std_mm_h_sqrt_h: float = 0.0,
    process_noise_sigma_km: float = 1.0,
    random_seed: int | None = None,
) -> RainField:
    """Convenience wrapper for the advection-diffusion simulator."""

    return simulate_advection_diffusion_rain(
        AdvectionDiffusionConfig(
            grid=GridSpec(
                domain_width_km=domain_width_km,
                domain_height_km=domain_height_km,
                dx_km=dx_km,
                dy_km=dy_km,
            ),
            time=TimeSpec(duration_h=duration_h, dt_h=dt_h),
            wind_velocity_km_h=wind_velocity_km_h,
            wind_field_km_h=wind_field_km_h,
            diffusion_km2_h=diffusion_km2_h,
            decay_h_inv=decay_h_inv,
            process_noise_std_mm_h_sqrt_h=process_noise_std_mm_h_sqrt_h,
            process_noise_sigma_km=process_noise_sigma_km,
            random_seed=random_seed,
        )
    )


def simulate_gaussian_rain_field(
    *,
    domain_width_km: float = 30.0,
    domain_height_km: float = 30.0,
    dx_km: float = 0.5,
    dy_km: float = 0.5,
    duration_h: float = 1.0,
    dt_h: float = 1.0 / 60.0,
    wind_velocity_km_h: tuple[float, float] = (8.0, 3.0),
    decay_h_inv: float = 0.02,
    diffusion_km2_h: float = 0.04,
    seed: int | None = None,
    **_legacy_kwargs,
) -> RainField:
    """Legacy alias; now returns an advection-diffusion RainField."""

    rain_field = simulate_rain_field(
        domain_width_km=domain_width_km,
        domain_height_km=domain_height_km,
        dx_km=dx_km,
        dy_km=dy_km,
        duration_h=duration_h,
        dt_h=dt_h,
        wind_velocity_km_h=wind_velocity_km_h,
        diffusion_km2_h=diffusion_km2_h,
        decay_h_inv=decay_h_inv,
        random_seed=seed,
    )
    if seed is None:
        return rain_field
    return RainField(
        values_mm_h=rain_field.values_mm_h,
        grid=rain_field.grid,
        time=rain_field.time,
        model_name=rain_field.model_name,
        units=rain_field.units,
        random_seed=seed,
        model_params=rain_field.model_params,
        metadata=rain_field.metadata,
    )


if __name__ == "__main__":
    rain = simulate_rain_field()
    print("RainField summary:", rain.summary())
    print("Example R(10 km, 10 km, 0.5 h):", rain.sample_nearest(10.0, 10.0, 0.5))
