"""
accel_live_dash.py - SIT225 5C: smooth live Plotly Dash for smartphone accelerometer data.

Phone (Arduino IoT Remote) -> Arduino IoT Cloud -> this script -> smooth_live_dash -> browser

Run:
    python accel_live_dash.py                      # live data from the phone
    python accel_live_dash.py --replay data/x.csv  # replay a recording (no phone needed)
Then open http://127.0.0.1:8050
"""
import argparse
import os
import time

from phone_stream import CsvRecorder, start_csv_replay, start_phone_stream
from smooth_dash import smooth_live_dash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--window", type=float, default=20, help="seconds visible on screen")
    ap.add_argument("--overlay", action="store_true", help="x, y, z on one plot instead of 3 subplots")
    ap.add_argument("--replay", help="replay a recorded CSV instead of connecting to the cloud")
    ap.add_argument("--no-interp", action="store_true", help="disable leading-edge interpolation")
    args = ap.parse_args()

    # This one call sets up the whole dashboard: buffer, server callback and render loop.
    live = smooth_live_dash(
        ["x", "y", "z"],
        title="Smartphone accelerometer - smooth live view",
        y_title="m/s²",
        window_s=args.window,
        subplots=not args.overlay,
        min_y_span=1.0,        # a resting phone is nearly flat, so do not magnify the noise
        interpolate=not args.no_interp,
    )

    if args.replay:
        start_csv_replay(args.replay, live.push)     # push() is the callback, no glue needed
    else:
        path = os.path.join("data", time.strftime("accel_%Y%m%d_%H%M%S.csv"))
        recorder = CsvRecorder(path)
        print(f"[accel] recording to {path}")

        # Called from the Arduino Cloud thread every time one axis is updated.
        def on_sample(axis, value, t):
            live.push(axis, value, t)      # feeds the dashboard
            recorder.add(axis, value, t)   # and the CSV file

        start_phone_stream(on_sample)

    live.run(port=args.port)               # blocks here until Ctrl+C


if __name__ == "__main__":
    main()
