"""Tests for the packaged advection-diffusion rain simulator."""

import unittest

import numpy as np

from rain import (
    AdvectionDiffusionConfig,
    GaussianInitialCondition,
    GridSpec,
    TimeSpec,
    simulate_advection_diffusion_rain,
)
from rain.advection_diffusion import build_initial_field


class AdvectionDiffusionRainTest(unittest.TestCase):
    def test_returns_cell_centered_rain_field(self):
        rain = simulate_advection_diffusion_rain(_small_config())

        self.assertEqual(rain.values_mm_h.shape, (7, 12, 10))
        self.assertEqual(rain.grid.nx, 10)
        self.assertEqual(rain.grid.ny, 12)
        self.assertEqual(rain.model_name, "advection_diffusion")
        self.assertGreaterEqual(float(rain.values_mm_h.min()), 0.0)

    def test_mass_is_conserved_without_decay_or_diffusion_loss(self):
        config = _small_config(
            wind_velocity_km_h=(1.0, 0.5),
            diffusion_km2_h=0.02,
            decay_h_inv=0.0,
        )
        rain = simulate_advection_diffusion_rain(config)

        np.testing.assert_allclose(
            rain.values_mm_h[0].sum(),
            rain.values_mm_h[-1].sum(),
            rtol=1e-12,
        )

    def test_positive_x_wind_moves_center_of_mass_east(self):
        config = _small_config(
            wind_velocity_km_h=(1.0, 0.0),
            diffusion_km2_h=0.0,
            decay_h_inv=0.0,
        )
        rain = simulate_advection_diffusion_rain(config)

        self.assertGreater(
            _x_center_of_mass(rain.grid, rain.values_mm_h[-1]),
            _x_center_of_mass(rain.grid, rain.values_mm_h[0]),
        )

    def test_seeded_process_noise_is_reproducible(self):
        config = _small_config(
            process_noise_std_mm_h_sqrt_h=0.5,
            process_noise_sigma_km=1.0,
            random_seed=123,
        )

        rain_a = simulate_advection_diffusion_rain(config)
        rain_b = simulate_advection_diffusion_rain(config)

        np.testing.assert_allclose(rain_a.values_mm_h, rain_b.values_mm_h)

    def test_process_noise_changes_the_trajectory(self):
        deterministic = simulate_advection_diffusion_rain(_small_config())
        noisy = simulate_advection_diffusion_rain(
            _small_config(
                process_noise_std_mm_h_sqrt_h=0.5,
                process_noise_sigma_km=1.0,
                random_seed=123,
            )
        )

        self.assertGreater(
            float(np.abs(deterministic.values_mm_h - noisy.values_mm_h).sum()),
            0.0,
        )

    def test_larger_process_noise_sigma_is_smoother(self):
        rough = simulate_advection_diffusion_rain(
            _small_config(
                process_noise_std_mm_h_sqrt_h=1.0,
                process_noise_sigma_km=0.0,
                random_seed=123,
            )
        )
        smooth = simulate_advection_diffusion_rain(
            _small_config(
                process_noise_std_mm_h_sqrt_h=1.0,
                process_noise_sigma_km=2.0,
                random_seed=123,
            )
        )

        self.assertLess(_roughness(smooth.values_mm_h[1]), _roughness(rough.values_mm_h[1]))

    def test_gaussian_initial_condition_uses_full_covariance_matrix(self):
        grid = GridSpec(
            domain_width_km=20.0,
            domain_height_km=20.0,
            dx_km=0.5,
            dy_km=0.5,
        )
        condition = GaussianInitialCondition(
            mu_km=np.array([10.0, 10.0]),
            covariance_km2=np.array([[4.0, 1.5], [1.5, 2.0]]),
            intensity_mm_h=10.0,
        )

        field = build_initial_field(grid, (condition,))

        self.assertGreater(_weighted_xy_covariance(grid, field), 0.0)


def _small_config(
    *,
    wind_velocity_km_h=(1.0, 0.0),
    diffusion_km2_h=0.0,
    decay_h_inv=0.0,
    process_noise_std_mm_h_sqrt_h=0.0,
    process_noise_sigma_km=1.0,
    random_seed=None,
) -> AdvectionDiffusionConfig:
    return AdvectionDiffusionConfig(
        grid=GridSpec(
            domain_width_km=10.0,
            domain_height_km=12.0,
            dx_km=1.0,
            dy_km=1.0,
        ),
        time=TimeSpec(duration_h=0.6, dt_h=0.1),
        wind_velocity_km_h=wind_velocity_km_h,
        diffusion_km2_h=diffusion_km2_h,
        decay_h_inv=decay_h_inv,
        process_noise_std_mm_h_sqrt_h=process_noise_std_mm_h_sqrt_h,
        process_noise_sigma_km=process_noise_sigma_km,
        random_seed=random_seed,
        initial_conditions=(
            GaussianInitialCondition(
                mu_km=np.array([3.0, 6.0]),
                covariance_km2=np.array([[1.0, 0.0], [0.0, 1.0]]),
                intensity_mm_h=10.0,
            ),
        ),
    )


def _x_center_of_mass(grid: GridSpec, field: np.ndarray) -> float:
    weights_by_x = field.sum(axis=0)
    return float(np.dot(grid.x_centers_km, weights_by_x) / weights_by_x.sum())


def _roughness(field: np.ndarray) -> float:
    return float(
        np.mean(np.diff(field, axis=0) ** 2)
        + np.mean(np.diff(field, axis=1) ** 2)
    )


def _weighted_xy_covariance(grid: GridSpec, field: np.ndarray) -> float:
    xx, yy = np.meshgrid(grid.x_centers_km, grid.y_centers_km)
    total = field.sum()
    mean_x = float((xx * field).sum() / total)
    mean_y = float((yy * field).sum() / total)
    return float((((xx - mean_x) * (yy - mean_y) * field).sum()) / total)


if __name__ == "__main__":
    unittest.main()
