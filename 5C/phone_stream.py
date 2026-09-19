"""
phone_stream.py - receive the phone's accelerometer x, y, z from Arduino IoT Cloud.

The Python "device" Thing has three cloud variables (python_x, python_y, python_z) that are
synced with the phone Thing's accelerometer_x/_y/_z. Every time the cloud writes one of
them, `on_sample(axis, value, t)` is called from a background thread.

Credentials are read from cloud_credentials.py (not committed) or from the environment
variables ARDUINO_DEVICE_ID / ARDUINO_SECRET_KEY.
"""
import csv
import logging
import os
import threading
import time

VARIABLES = {"x": "python_x", "y": "python_y", "z": "python_z"}


def load_credentials():
    """Read the device id and secret key, which are never committed to git."""
    try:
        from cloud_credentials import DEVICE_ID, SECRET_KEY
    except ImportError:
        DEVICE_ID = os.environ.get("ARDUINO_DEVICE_ID")     # fall back to the environment
        SECRET_KEY = os.environ.get("ARDUINO_SECRET_KEY")

    if not DEVICE_ID or not SECRET_KEY:
        raise SystemExit("Missing credentials: create cloud_credentials.py with DEVICE_ID and SECRET_KEY")
    return DEVICE_ID, SECRET_KEY


def start_phone_stream(on_sample, variables=VARIABLES):
    """Connect to Arduino IoT Cloud in a daemon thread. Returns the thread."""
    from arduino_iot_cloud import ArduinoCloudClient

    device_id, secret_key = load_credentials()

    def worker():
        # The library needs the device id as the username too, otherwise the broker
        # rejects the connection with "index out of range" or error code 5.
        client = ArduinoCloudClient(device_id=device_id, username=device_id,
                                    password=secret_key, sync_mode=True)   # sync: we drive the loop

        # One callback per axis. The default argument pins `axis`, otherwise all three
        # lambdas would end up sharing the last value of the loop variable.
        for axis, var in variables.items():
            client.register(var, value=None,
                            on_write=lambda _c, value, axis=axis: on_sample(axis, value, time.time()))

        print("[phone_stream] connecting to Arduino IoT Cloud ...")
        client.start()                       # blocks until the Thing is discovered
        print("[phone_stream] connected, waiting for accelerometer data")

        while True:
            try:
                client.update()              # reads one MQTT message, waits at most 50 ms
            except Exception as e:
                logging.warning(f"[phone_stream] {e}; retrying")
                time.sleep(1.0)              # update() reconnects by itself on the next call

    th = threading.Thread(target=worker, name="arduino-cloud", daemon=True)
    th.start()
    return th


def start_csv_replay(path, on_sample, speed=1.0, loop=True):
    """Replay a recorded CSV (timestamp,x,y,z) in real time - handy for testing without the phone."""
    def worker():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f)]
        if not rows:
            return

        while True:
            t_first = float(rows[0]["timestamp"])   # first timestamp in the file
            start = time.time()                     # the same instant, but now

            for r in rows:
                due = start + (float(r["timestamp"]) - t_first) / speed   # keep the original spacing
                time.sleep(max(0.0, due - time.time()))
                now = time.time()
                for axis in ("x", "y", "z"):
                    on_sample(axis, float(r[axis]), now)

            if not loop:
                return

    th = threading.Thread(target=worker, name="csv-replay", daemon=True)
    th.start()
    return th


class CsvRecorder:
    """Writes one line <timestamp>,<x>,<y>,<z> each time all three axes have been updated."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(["timestamp", "x", "y", "z"])
        self._latest = {}                 # axis -> (time, value) collected so far
        self._lock = threading.Lock()     # the cloud thread and the replay thread both call add()
        self.rows = 0

    def add(self, axis, value, t):
        with self._lock:
            self._latest[axis] = (t, float(value))

            # The cloud sends the three axes separately, so wait until a full set is here.
            if len(self._latest) == 3:
                t_row = max(v[0] for v in self._latest.values())   # stamp the row with the last arrival
                self._w.writerow([f"{t_row:.3f}"] + [self._latest[a][1] for a in ("x", "y", "z")])
                self._f.flush()               # flush now, the script is usually stopped with Ctrl+C
                self._latest.clear()
                self.rows += 1
