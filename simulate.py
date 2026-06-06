#!/usr/bin/env python3
"""Create a simple map of links and weather stations."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create links + station map (minimal version).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size-km", type=float, default=30.0)
    parser.add_argument("--links", type=int, default=40)
    parser.add_argument("--rg", type=int, default=8, help="Number of rain gauges")
    parser.add_argument("--pws", type=int, default=120, help="Number of personal weather stations")
    parser.add_argument("--len-min", type=float, default=0.5)
    parser.add_argument("--len-max", type=float, default=8.0)
    parser.add_argument("--len-fixed", type=float, default=3.0)
    parser.add_argument("--random-locations", action="store_true", default=True)
    parser.add_argument("--fixed-locations", action="store_true")
    parser.add_argument("--random-lengths", action="store_true", default=True)
    parser.add_argument("--fixed-lengths", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def random_point(size_km: float) -> tuple[float, float]:
    return random.uniform(0.0, size_km), random.uniform(0.0, size_km)


def fixed_point(index: int, total: int, size_km: float) -> tuple[float, float]:
    cols = int(math.ceil(math.sqrt(total)))
    rows = int(math.ceil(total / cols))
    c = index % cols
    r = index // cols
    x = ((c + 1) / (cols + 1)) * size_km
    y = ((r + 1) / (rows + 1)) * size_km
    return x, y


def deploy_links(
    count: int,
    size_km: float,
    random_locations: bool,
    random_lengths: bool,
    len_min: float,
    len_max: float,
    len_fixed: float,
) -> list[dict]:
    links = []
    for i in range(count):
        if random_locations:
            x0, y0 = random_point(size_km)
            angle = random.uniform(0.0, 2.0 * math.pi)
        else:
            x0, y0 = fixed_point(i, count, size_km)
            angle = 0.0
        length = random.uniform(len_min, len_max) if random_lengths else len_fixed
        x1 = min(max(x0 + length * math.cos(angle), 0.0), size_km)
        y1 = min(max(y0 + length * math.sin(angle), 0.0), size_km)
        links.append(
            {
                "id": i,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "length_km": math.hypot(x1 - x0, y1 - y0),
            }
        )
    return links


def deploy_stations(rg_count: int, pws_count: int, size_km: float, random_locations: bool) -> list[dict]:
    stations = []
    total = rg_count + pws_count
    for i in range(total):
        x, y = random_point(size_km) if random_locations else fixed_point(i, total, size_km)
        kind = "RG" if i < rg_count else "PWS"
        sid = i if kind == "RG" else 10000 + (i - rg_count)
        stations.append({"id": sid, "x": x, "y": y, "kind": kind})
    return stations


def save_links_csv(path: Path, links: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["link_id", "x0_km", "y0_km", "x1_km", "y1_km", "length_km"])
        for lk in links:
            writer.writerow([lk["id"], lk["x0"], lk["y0"], lk["x1"], lk["y1"], lk["length_km"]])


def save_stations_csv(path: Path, stations: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["station_id", "kind", "x_km", "y_km"])
        for st in stations:
            writer.writerow([st["id"], st["kind"], st["x"], st["y"]])


def plot_map(path: Path, size_km: float, links: list[dict], stations: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    for lk in links:
        ax.plot([lk["x0"], lk["x1"]], [lk["y0"], lk["y1"]], "k-", lw=1.1, alpha=0.8)
    for st in stations:
        if st["kind"] == "RG":
            ax.plot(st["x"], st["y"], "s", color="darkgreen", ms=7)
        else:
            ax.plot(st["x"], st["y"], ".", color="orange", ms=4, alpha=0.7)
    ax.set_xlim(0.0, size_km)
    ax.set_ylim(0.0, size_km)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Links + Weather Stations")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    random_locations = not args.fixed_locations
    random_lengths = not args.fixed_lengths

    links = deploy_links(
        count=args.links,
        size_km=args.size_km,
        random_locations=random_locations,
        random_lengths=random_lengths,
        len_min=args.len_min,
        len_max=args.len_max,
        len_fixed=args.len_fixed,
    )
    stations = deploy_stations(
        rg_count=args.rg,
        pws_count=args.pws,
        size_km=args.size_km,
        random_locations=random_locations,
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    save_links_csv(out_dir / "links.csv", links)
    save_stations_csv(out_dir / "stations.csv", stations)
    plot_map(out_dir / "deployment_map.png", args.size_km, links, stations)

    print("Done.")
    print(f"Saved in: {out_dir.resolve()}")
    print("- links.csv")
    print("- stations.csv")
    print("- deployment_map.png")


if __name__ == "__main__":
    main()
