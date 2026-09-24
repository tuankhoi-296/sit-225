# SIT225 5C – Smooth live dashboard for phone accelerometer data

My phone streams its accelerometer into Arduino IoT Cloud, a Python script picks the values up,
and Plotly Dash draws them live.

The Week 8 version waited for N samples and then redrew the whole figure, so the graph sat still
and then jumped. Here I draw the data about a second behind real time and let the browser redraw
at 30 fps, the same idea a video player uses when it buffers.

## Files

- `smooth_dash.py` – `smooth_live_dash()`, the wrapper that does the smoothing for any data
- `accel_live_dash.py` – the program I run: cloud → dashboard, and records a CSV
- `phone_stream.py` – Arduino IoT Cloud connection, CSV recorder, CSV replay
- `baseline_naive_dash.py` – the Week 8 approach, for comparison
- `demo_any_stream.py` – same wrapper on CPU usage and a fake signal
- `plot_recorded.py` – static graphs from a recording
- `data/`, `report/figures/` – recordings, graphs, report figures

## Setup

```
pip install dash plotly pandas numpy matplotlib arduino-iot-cloud psutil
```

Credentials go in `cloud_credentials.py`, which is git-ignored:

```python
DEVICE_ID = "<device id of the Python device>"
SECRET_KEY = "<secret key>"
```

The Arduino side needs a Python device with `python_x`, `python_y`, `python_z` synced to the
phone's `accelerometer_x/y/z`.

Two things cost me time: the client needs the device id as the **username** as well as the
password, otherwise you get a confusing "index out of range"; and only **one** program can hold a
Device ID at a time, so two running scripts will keep kicking each other off with `[Errno 10054]`.

## Running

```
python accel_live_dash.py                              # live, http://127.0.0.1:8050
python accel_live_dash.py --replay data/accel_....csv  # replay, no phone needed
python baseline_naive_dash.py                          # Week 8 version, http://127.0.0.1:8052
python demo_any_stream.py                              # other data, http://127.0.0.1:8051
python plot_recorded.py data/accel_....csv
```

## Using the wrapper

Name your signals, then call `push()` whenever a value turns up:

```python
from smooth_dash import smooth_live_dash

live = smooth_live_dash(["x", "y", "z"], title="Accelerometer", y_title="m/s²")
live.start()                        # background thread; live.run() blocks instead

live.push("x", 0.12)                # timestamped now
live.push("y", 0.30, t=1789305071)  # or with the sensor's own Unix time
live.push_row({"x": .1, "y": .3, "z": 9.8})
```

`push()` works from any thread and drops NaN or anything that isn't a number.

`channels` is the only required argument. The useful options are `window_s` (20 s on screen),
`delay_s` (0.6 s, grows on its own if data arrives late), `fps` (30), `interpolate` (True, grows
the line towards the next sample instead of jumping), and `subplots` (one plot per channel).
The others have defaults that worked for both the accelerometer and the demo.

## Limitations

The picture is about 1.5 s behind real time; that delay is what buys the smoothness. And
smoothing can't invent data, Arduino IoT Cloud only sent about one sample per second per axis,
and only when a value changed, so a phone sitting still gives a flat line.
