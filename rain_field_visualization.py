"""Visualization helpers for Gaussian rain-field simulations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display

from rain_field_simulation import RainField


def animate_rain_field(
    rain: RainField,
    *,
    interval_ms: int = 120,
    cmap: str = "Blues",
    vmin: float = 0.0,
    vmax: float | None = None,
    title: str = "Simulated rain field",
) -> FuncAnimation:
    """Create a Matplotlib animation for a RainField.

    Args:
        rain: RainField returned by simulate_gaussian_rain_field.
        interval_ms: Delay between frames in milliseconds.
        cmap: Matplotlib colormap name.
        vmin: Minimum color scale value in mm/hour.
        vmax: Maximum color scale value in mm/hour. Defaults to field maximum.
        title: Plot title prefix.

    Returns:
        A FuncAnimation object. In notebooks, pass it to display_rain_animation.
    """

    if vmax is None:
        vmax = float(rain.values_mm_h.max())

    extent = [
        float(rain.x_km.min()),
        float(rain.x_km.max()),
        float(rain.y_km.min()),
        float(rain.y_km.max()),
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(
        rain.values_mm_h[0],
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        animated=True,
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Rain rate [mm/hour]")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")

    def update(frame_index: int):
        image.set_array(rain.values_mm_h[frame_index])
        ax.set_title(f"{title}, t = {rain.t_h[frame_index]:.2f} h")
        return (image,)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(rain.t_h),
        interval=interval_ms,
        blit=False,
    )
    plt.close(fig)
    return animation


def display_rain_animation(animation: FuncAnimation) -> HTML:
    """Display a rain animation inline in a Jupyter notebook."""

    html = HTML(animation.to_jshtml())
    display(html)
    


def save_rain_animation(
    animation: FuncAnimation,
    output_path: str | Path,
    *,
    fps: int = 10,
) -> Path:
    """Save a rain animation to disk.

    Use a ``.gif`` path for a Pillow-based GIF, or ``.mp4`` if ffmpeg is
    installed in the environment.
    """

    output_path = Path(output_path)
    if output_path.suffix.lower() == ".gif":
        animation.save(output_path, writer="pillow", fps=fps)
    else:
        animation.save(output_path, fps=fps)
    return output_path
