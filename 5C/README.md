# SIT225 5C – Smooth live Plotly Dash for smartphone accelerometer data

| File | Purpose |
|---|---|
| `smooth_dash.py` | **The reusable wrapper** – `smooth_live_dash(channels, ...)` |
| `accel_live_dash.py` | Phone → Arduino IoT Cloud → Python → smooth dashboard (+ CSV recording) |
| `phone_stream.py` | Arduino IoT Cloud connection, CSV replay, CSV recorder |
| `baseline_naive_dash.py` | Week 8 approach (redraw every N samples) for comparison |
| `demo_any_stream.py` | Same wrapper with unrelated data (CPU %, bursty synthetic signal) |
| `plot_recorded.py` | Static PNG graphs from a recorded CSV |
| `data/` | Recorded CSV files and generated graphs |
| `report/` | Figures and report builder |

## Setup

```
pip install dash plotly pandas numpy matplotlib arduino-iot-cloud psutil
```

Create `cloud_credentials.py` (git-ignored):

```python
DEVICE_ID = "<device id of the Python device>"
SECRET_KEY = "<secret key>"
```

The Python Thing needs variables `python_x`, `python_y`, `python_z` synced with the phone's
`accelerometer_x/y/z`. Only **one** program may use a Device ID at a time.

## Run

```
python accel_live_dash.py                 # live, open http://127.0.0.1:8050
python accel_live_dash.py --replay data/accel_....csv
python baseline_naive_dash.py             # comparison, http://127.0.0.1:8052
python demo_any_stream.py                 # other data, http://127.0.0.1:8051
python plot_recorded.py data/accel_....csv
```

## `smooth_live_dash` API

```python
from smooth_dash import smooth_live_dash

live = smooth_live_dash(["x", "y", "z"], title="Accelerometer", y_title="m/s²")
live.start()                        # Dash server in a background thread (or live.run() to block)

live.push("x", 0.12)                # one value, timestamped now
live.push("y", 0.30, t=1789305071)  # with the sensor's own Unix timestamp (seconds)
live.push_row({"x": .1, "y": .3, "z": 9.8})
```

| Option | Default | Meaning |
|---|---|---|
| `channels` | – | signal names; one line (and one subplot) each |
| `title` | `"Live data"` | page heading |
| `window_s` | `20` | seconds visible on screen |
| `delay_s` | `0.6` | minimum playout delay; grows automatically when data arrives late |
| `max_delay_s` | `5` | upper bound for the adaptive delay |
| `fps` | `30` | browser redraw rate |
| `poll_ms` | `150` | how often the browser fetches new samples |
| `interpolate` | `True` | grow lines towards the next sample every frame (smooth for ~1 Hz data) |
| `max_gap_s` | `2.5` | no interpolation across longer gaps |
| `subplots` | `True` | `False` = all channels on one plot |
| `y_title`, `y_range`, `min_y_span`, `colors`, `line_width`, `height` | | appearance (`y_range` fixes the axis) |
| `max_points` | `3000` | per-channel cap of points kept in the browser |
| `buffer_capacity` | `50000` | samples kept on the server |
| `name` | `"live"` | id prefix – use different names for several dashboards in one app |
| `app` | `None` | existing `dash.Dash` app; then place `live.component` in your own layout |

Returned object: `.push()`, `.push_row()`, `.start()`, `.run()`, `.app`, `.component`, `.buffer`.
`push()` is thread-safe, and ignores NaN or values that are not numbers.
