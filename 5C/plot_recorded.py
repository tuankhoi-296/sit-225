"""
plot_recorded.py - static graphs of a recorded accelerometer CSV (timestamp,x,y,z).

Run:  python plot_recorded.py data/accel_YYYYmmdd_HHMMSS.csv
Saves <csv name>_xyz.png (3 subplots) and <csv name>_overlay.png next to the CSV.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"x": "#2a78d6", "y": "#e8543a", "z": "#1fa37a"}


def main(path):
    path = Path(path)
    df = pd.read_csv(path)

    # The CSV stores Unix seconds, which would show as UTC on the axis.
    df["time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Australia/Melbourne")
    fmt = mdates.DateFormatter("%H:%M:%S", tz=df["time"].dt.tz)

    # First figure: one subplot per axis, sharing the time axis.
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for ax, axis in zip(axes, ["x", "y", "z"]):
        ax.plot(df["time"], df[axis], color=COLORS[axis], lw=1.2)
        ax.set_ylabel(f"{axis} (m/s²)")
        ax.grid(alpha=0.3)
    axes[-1].xaxis.set_major_formatter(fmt)
    axes[0].set_title(f"Smartphone accelerometer - {path.name} ({len(df)} rows)")
    fig.tight_layout()
    out1 = path.with_name(path.stem + "_xyz.png")
    fig.savefig(out1, dpi=150)

    # Second figure: all three on one pair of axes, easier to compare.
    fig, ax = plt.subplots(figsize=(11, 4))
    for axis in ["x", "y", "z"]:
        ax.plot(df["time"], df[axis], color=COLORS[axis], lw=1.2, label=axis)
    ax.xaxis.set_major_formatter(fmt)
    ax.set_ylabel("m/s²")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_title("x, y, z overlay")
    fig.tight_layout()
    out2 = path.with_name(path.stem + "_overlay.png")
    fig.savefig(out2, dpi=150)
    print(f"saved {out1}\nsaved {out2}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python plot_recorded.py data/accel_....csv")
    main(sys.argv[1])
