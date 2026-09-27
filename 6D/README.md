# SIT225 6D – Annotating accelerometer data with webcam activity images

Phone (Arduino IoT Remote) → Arduino IoT Cloud → Python → 10-second windows of accelerometer
data + a webcam image per window → labels from the images → pattern analysis.

| File                     | Purpose                                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `capture_dash.py`      | Capture session: cloud stream + webcam, one CSV/JPG/PNG per 10 s window, live Plotly Dash page with the activity cue |
| `phone_stream.py`      | Arduino IoT Cloud client for`python_x/y/z` (reused from 5C)                                                        |
| `annotate.py`          | Label each window from its images only →`data/annotations.csv`                                                    |
| `analyze.py`           | Features, figures, pair contrasts, decision trees, resolution of uncertain windows                                   |
| `data/windows/`        | `<seq>_<yyyymmddHHMMss>.csv` (timestamp,x,y,z), `.jpg` (webcam), `.png` (graph)                                |
| `data/strips/`         | 4 webcam frames per window, side by side (annotation aid)                                                            |
| `data/session_log.csv` | filename, seq, window start/end, rows, cue                                                                           |
| `data/raw_session.csv` | every value received: timestamp, axis, value                                                                         |
| `figures/`             | Figures used in the report                                                                                           |

The report itself (`SIT225_6D_Report.pdf`) and the script that builds it are submitted on OnTrack and
are not kept in this repository.

## Setup

```
pip install dash plotly pandas numpy matplotlib scikit-learn opencv-python arduino-iot-cloud python-docx
```

Create `cloud_credentials.py` (git-ignored):

```python
DEVICE_ID = "<device id of the Python device>"
SECRET_KEY = "<secret key>"
```

## Run

```
python capture_dash.py                      # 35 min session, http://127.0.0.1:8050
python capture_dash.py --minutes 2 --lead 10 --out data_test   # short test
python annotate.py                          # label the windows (keys 0/1/2, u, b, q)
python analyze.py                           # features, figures, data/results.json
```

`capture_dash.py` options: `--minutes`, `--window` (seconds per window), `--block` (windows per
activity), `--lead` (preparation countdown), `--out`, `--port`, `--seed`.

## Recorded session

35 minutes, 210 windows, 3 activities in shuffled 30 s blocks. 1156 accelerometer rows arrived
(mean 5.5 per window; 40 windows empty because of gaps in the Arduino Cloud stream), leaving 136
usable windows. Labels from images agree with the cue schedule 78.7 % of the time; a decision tree
on the sensor features reaches 97.8 % against the cue schedule. See the report for the analysis.
