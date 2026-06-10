"""Show the minimal state-space rain simulation.

Run from the project root:

    python3 simulation/show_state_space_rain.py

Optional save:

    python3 simulation/show_state_space_rain.py --save simulation/state_space_rain.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.state_space_rain import (
    StateGrid,
    build_transition_matrix,
    gaussian_initial_field,
    simulate_state_space_rain,
)


def main() -> None:
    args = parse_args()

    grid = StateGrid(nx=80, ny=80, dx_km=0.25, dy_km=0.25)
    initial_field = gaussian_initial_field(
        grid,
        center_x_km=6.0,
        center_y_km=10.0,
        sigma_x_km=1.2,
        sigma_y_km=1.8,
        intensity_mm_h=35.0,
    )

    transition = build_transition_matrix(
        grid,
        dt_h=1.0 / 60.0,
        wind_velocity_km_h=(8.0, 2.5),
        diffusion_km2_h=0.04,
        decay_h_inv=0.02,
    )
    

    rain = simulate_state_space_rain(
        grid=grid,
        initial_field_mm_h=initial_field,
        transition_matrix=transition,
        num_steps=150,
    )

    animation = make_animation(grid, rain)

    if args.save is not None:
        save_path = Path(args.save)
        if save_path.suffix.lower() == ".gif":
            animation.save(save_path, writer="pillow", fps=12)
        else:
            animation.save(save_path, fps=12)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()


def make_animation(grid: StateGrid, rain):
    """Create a Matplotlib animation for R[t, y, x]."""

    extent = [
        float(grid.x_km.min()),
        float(grid.x_km.max()),
        float(grid.y_km.min()),
        float(grid.y_km.max()),
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(
        rain[0],
        origin="lower",
        extent=extent,
        cmap="Blues",
        vmin=0.0,
        vmax=float(rain.max()),
        animated=True,
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Rain rate [mm/hour]")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")

    def update(frame_index: int):
        image.set_array(rain[frame_index])
        ax.set_title(f"State-space rain simulation, step {frame_index}")
        return (image,)

    return FuncAnimation(
        fig,
        update,
        frames=rain.shape[0],
        interval=80,
        blit=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output path, for example simulation/state_space_rain.gif",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
