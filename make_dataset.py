#!/usr/bin/env python3
"""
Full pipeline: sensor layout (Ron) + rain physics (Nir) -> simulated dataset.

Steps:
1. Deploy hub-based links and stations (reuses june26/simulate.py functions)
2. Simulate rain field R(x,y,t) with Nir/state_space_rain.py
3. Sample the rain at every sensor for every time step
4. Save dataset to june26/outputs_dataset/

Outputs:
- links.csv / stations.csv         (geometry; links now include freq_ghz, pol)
- links_measurements.csv           (per link/time: true path rain + attenuation in dB)
- stations_rain.csv                (time, station_id, kind, rain mm/h)
- rain_field.npy                   (true field R[t, y, x] for later use)
- snapshot.png                     (rain + sensors at the middle time step)

Link measurement model (doc): A = a*R^b per km, integrated along the path,
plus baseline and noise, then quantized:
    A_rain = mean_over_path( a * R^b ) * L
    A_tot  = A0 + A_rain + W,   W ~ N(0, sigma^2)
    A_obs  = Q(A_tot)
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Make both sub-projects importable
sys.path.insert(0, str(SCRIPT_DIR))          # june26/simulate.py
sys.path.insert(0, str(PROJECT_DIR / "Nir"))  # Nir/state_space_rain.py

from simulate import deploy_links, deploy_stations, save_links_csv, save_stations_csv
from simulation.state_space_rain import (
    StateGrid,
    build_transition_matrix,
    gaussian_initial_field,
    simulate_state_space_rain,
)

# ITU-R P.838 power-law coefficients: specific attenuation gamma = k * R^alpha
# (dB/km), with R in mm/h. Keys are (frequency_ghz, polarization).
ITU_R = {
    (5.0, "H"): (0.0000387, 0.912),
    (5.0, "V"): (0.0000356, 0.930),
    (20.0, "H"): (0.00713, 1.061),
    (20.0, "V"): (0.00665, 1.077),
    (38.0, "H"): (0.0758, 1.099),
    (38.0, "V"): (0.0713, 1.113),
    (70.0, "H"): (0.492, 1.115),
    (70.0, "V"): (0.465, 1.121),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full simulated rain dataset.")

    g_layout = parser.add_argument_group("sensor layout (Ron)")
    g_layout.add_argument("--seed", type=int, default=42)
    g_layout.add_argument("--size-km", type=float, default=30.0)
    g_layout.add_argument("--hubs", type=int, default=14)
    g_layout.add_argument("--spokes-min", type=int, default=1)
    g_layout.add_argument("--spokes-max", type=int, default=6)
    g_layout.add_argument("--len-min", type=float, default=0.5)
    g_layout.add_argument("--len-max", type=float, default=8.0)
    g_layout.add_argument("--rg", type=int, default=8)
    g_layout.add_argument("--pws", type=int, default=120)

    g_link = parser.add_argument_group("link measurement model (power-law)")
    g_link.add_argument("--link-a0-db", type=float, default=0.0, help="Baseline dry attenuation A0 (dB)")
    g_link.add_argument("--link-noise-db", type=float, default=0.5, help="Std of W ~ N(0, sigma^2) (dB)")
    g_link.add_argument("--link-quant-db", type=float, default=0.1, help="Quantization step Q (dB); 0 disables")

    g_rain = parser.add_argument_group("rain physics (Nir)")
    g_rain.add_argument("--cell-km", type=float, default=1.0, help="Grid cell size")
    g_rain.add_argument("--wind-x", type=float, default=6.0, help="Wind east-west (km/h)")
    g_rain.add_argument("--wind-y", type=float, default=3.0, help="Wind south-north (km/h)")
    g_rain.add_argument("--diffusion", type=float, default=0.5, help="Spread rate (km^2/h)")
    g_rain.add_argument("--decay", type=float, default=0.1, help="Rain decay rate (1/h)")
    g_rain.add_argument("--rain-center-x", type=float, default=10.0, help="Initial blob x (km)")
    g_rain.add_argument("--rain-center-y", type=float, default=10.0, help="Initial blob y (km)")
    g_rain.add_argument("--rain-sigma", type=float, default=4.0, help="Initial blob width (km)")
    g_rain.add_argument("--rain-intensity", type=float, default=25.0, help="Peak rain (mm/h)")

    g_time = parser.add_argument_group("time")
    g_time.add_argument("--dt-min", type=float, default=5.0, help="Time step (minutes)")
    g_time.add_argument("--steps", type=int, default=60, help="Number of time steps")

    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs_dataset")
    return parser.parse_args()


def sample_field_at_point(field: np.ndarray, x_km: float, y_km: float, cell_km: float) -> float:
    """Nearest-cell rain value at a point. field is one time slice (ny, nx)."""
    ny, nx = field.shape
    ix = min(max(int(x_km / cell_km), 0), nx - 1)
    iy = min(max(int(y_km / cell_km), 0), ny - 1)
    return float(field[iy, ix])


def link_path_rain_and_attenuation(
    field: np.ndarray, link: dict, cell_km: float, n_samples: int = 20
) -> tuple[float, float]:
    """
    Return (path_rain_mm_h, A_rain_db) for one link at one time.

    A_rain is the rain-induced attenuation from the ITU-R power law,
    integrated along the path:
        gamma(l) = k * R(l)^alpha          [dB/km]
        A_rain   = mean(gamma) * length    [dB]
    The power law is applied per sample BEFORE averaging, so the path
    nonlinearity is preserved (unlike averaging rain first).
    """
    k, alpha = ITU_R[(link["freq_ghz"], link["pol"])]
    rain_sum = 0.0
    gamma_sum = 0.0
    for i in range(n_samples):
        f = i / (n_samples - 1)
        x = link["x0"] + f * (link["x1"] - link["x0"])
        y = link["y0"] + f * (link["y1"] - link["y0"])
        r = sample_field_at_point(field, x, y, cell_km)
        rain_sum += r
        gamma_sum += k * (max(r, 0.0) ** alpha)
    path_rain = rain_sum / n_samples
    a_rain_db = (gamma_sum / n_samples) * link["length_km"]
    return path_rain, a_rain_db


def save_links_measurements_csv(
    path: Path,
    links: list[dict],
    path_rain: np.ndarray,
    a_tot_db: np.ndarray,
    time_min: np.ndarray,
) -> None:
    """One row per link per time: truth (path rain) + measurement (attenuation)."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["time_min", "link_id", "freq_ghz", "pol", "length_km", "path_rain_mm_h", "A_tot_db"]
        )
        for t_idx, t in enumerate(time_min):
            for i, lk in enumerate(links):
                writer.writerow(
                    [
                        float(t),
                        lk["id"],
                        lk["freq_ghz"],
                        lk["pol"],
                        lk["length_km"],
                        path_rain[i, t_idx],
                        a_tot_db[i, t_idx],
                    ]
                )


def save_stations_rain_csv(path: Path, stations: list[dict], rain: np.ndarray, time_min: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_min", "station_id", "kind", "rain_mm_h"])
        for t_idx, t in enumerate(time_min):
            for i, st in enumerate(stations):
                writer.writerow([float(t), st["id"], st["kind"], rain[i, t_idx]])


def plot_snapshot(
    path: Path, size_km: float, field: np.ndarray, links: list[dict], stations: list[dict], t_min: float
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        field, origin="lower", extent=[0, size_km, 0, size_km], cmap="Blues", alpha=0.8
    )
    fig.colorbar(im, ax=ax, label="rain (mm/h)", shrink=0.8)
    for lk in links:
        ax.plot([lk["x0"], lk["x1"]], [lk["y0"], lk["y1"]], "k-", lw=1.0, alpha=0.7)
    for st in stations:
        if st["kind"] == "RG":
            ax.plot(st["x"], st["y"], "s", color="darkgreen", ms=7)
        else:
            ax.plot(st["x"], st["y"], ".", color="orange", ms=4, alpha=0.7)
    ax.set_xlim(0, size_km)
    ax.set_ylim(0, size_km)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title(f"Rain field + sensors (t = {t_min:.0f} min)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)  # for measurement noise

    # --- 1) Sensor layout (Ron) ---
    links = deploy_links(
        hubs=args.hubs,
        spokes_min=args.spokes_min,
        spokes_max=args.spokes_max,
        size_km=args.size_km,
        random_locations=True,
        random_lengths=True,
        len_min=args.len_min,
        len_max=args.len_max,
        len_fixed=3.0,
    )
    stations = deploy_stations(
        rg_count=args.rg, pws_count=args.pws, size_km=args.size_km, random_locations=True
    )

    # --- 2) Rain physics (Nir) ---
    n_cells = int(round(args.size_km / args.cell_km))
    grid = StateGrid(nx=n_cells, ny=n_cells, dx_km=args.cell_km, dy_km=args.cell_km)
    initial = gaussian_initial_field(
        grid,
        center_x_km=args.rain_center_x,
        center_y_km=args.rain_center_y,
        sigma_x_km=args.rain_sigma,
        sigma_y_km=args.rain_sigma,
        intensity_mm_h=args.rain_intensity,
    )
    dt_h = args.dt_min / 60.0
    transition = build_transition_matrix(
        grid,
        dt_h=dt_h,
        wind_velocity_km_h=(args.wind_x, args.wind_y),
        diffusion_km2_h=args.diffusion,
        decay_h_inv=args.decay,
    )
    rain_cube = simulate_state_space_rain(
        grid=grid,
        initial_field_mm_h=initial,
        transition_matrix=transition,
        num_steps=args.steps,
    )  # shape: (steps + 1, ny, nx)

    # --- 3) Sample rain at sensors ---
    n_times = rain_cube.shape[0]
    time_min = np.arange(n_times) * args.dt_min

    links_path_rain = np.zeros((len(links), n_times))  # truth: mean rain along path
    links_a_rain = np.zeros((len(links), n_times))     # rain-induced attenuation (dB)
    stations_rain = np.zeros((len(stations), n_times))
    for t in range(n_times):
        field = rain_cube[t]
        for i, lk in enumerate(links):
            pr, a_rain = link_path_rain_and_attenuation(field, lk, args.cell_km)
            links_path_rain[i, t] = pr
            links_a_rain[i, t] = a_rain
        for i, st in enumerate(stations):
            stations_rain[i, t] = sample_field_at_point(field, st["x"], st["y"], args.cell_km)

    # --- 3b) Link measurement model: A_tot = A0 + A_rain + W, then quantize ---
    noise = rng.normal(0.0, args.link_noise_db, size=links_a_rain.shape)
    links_a_tot = args.link_a0_db + links_a_rain + noise
    links_a_tot = np.maximum(links_a_tot, 0.0)
    if args.link_quant_db > 0:
        links_a_tot = np.round(links_a_tot / args.link_quant_db) * args.link_quant_db

    # --- 4) Save everything ---
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    save_links_csv(out_dir / "links.csv", links)
    save_stations_csv(out_dir / "stations.csv", stations)
    save_links_measurements_csv(
        out_dir / "links_measurements.csv", links, links_path_rain, links_a_tot, time_min
    )
    save_stations_rain_csv(out_dir / "stations_rain.csv", stations, stations_rain, time_min)
    np.save(out_dir / "rain_field.npy", rain_cube)

    mid = n_times // 2
    plot_snapshot(out_dir / "snapshot.png", args.size_km, rain_cube[mid], links, stations, time_min[mid])

    print("Done.")
    print(f"Links: {len(links)}, stations: {len(stations)}, time steps: {n_times}")
    print(f"Duration: {time_min[-1]:.0f} minutes, dt = {args.dt_min:.0f} min")
    print(f"Link model: A0={args.link_a0_db} dB, noise={args.link_noise_db} dB, quant={args.link_quant_db} dB")
    print(f"Saved in: {out_dir.resolve()}")
    print("- links.csv / stations.csv      (geometry; links incl. freq_ghz, pol)")
    print("- links_measurements.csv        (path rain + attenuation A_tot in dB)")
    print("- stations_rain.csv             (rain at each station)")
    print("- rain_field.npy                (true field R[t,y,x])")
    print("- snapshot.png                  (visual check)")


if __name__ == "__main__":
    main()
