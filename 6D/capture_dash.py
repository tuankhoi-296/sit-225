"""
capture_dash.py - SIT225 6D: capture smartphone accelerometer data + webcam activity images.

Phone (Arduino IoT Remote) -> Arduino IoT Cloud -> this script
    every WINDOW seconds:
        data/windows/<seq>_<yyyymmddHHMMss>.csv   accelerometer rows (timestamp, x, y, z)
        data/windows/<seq>_<yyyymmddHHMMss>.jpg   webcam image taken when the window closes
        data/windows/<seq>_<yyyymmddHHMMss>.png   graph of the window
        data/strips/<seq>_<yyyymmddHHMMss>.jpg    4 frames spread over the window (annotation aid)
    -> Plotly Dash page (http://127.0.0.1:8050) shows the latest graph + image,
       and a cue telling the participant which activity to perform next.

The cue schedule (blocks of BLOCK windows, activity order shuffled every round) keeps the
three classes balanced. The cue is logged to data/session_log.csv but is NOT used for
annotation: labels are assigned later from the images only (annotate.py). The cue log is
used afterwards to measure how reliable image-based annotation was.

Run:
    python capture_dash.py                   # 35 minute session
    python capture_dash.py --minutes 2       # short test
"""
import argparse
import base64
import csv
import os
import random
import threading
import time
from datetime import datetime

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import dash
from dash import Input, Output, dcc, html
import plotly.graph_objects as go

from phone_stream import start_phone_stream

ACTIVITIES = {0: "no activity", 1: "waving", 2: "shaking"}
CUE_TEXT = {
    0: "KHONG LAM GI (no activity) - dat dien thoai nam tren ban, bo tay ra",
    1: "VAY TAY (waving) - cam dien thoai dung dung, vay tay qua lai",
    2: "LAC (shaking) - cam dien thoai nam ngang, lac nhanh len xuong",
}
SHORT = {0: "KHONG LAM GI", 1: "VAY TAY", 2: "LAC"}
COLORS = {"x": "#2a78d6", "y": "#e8543a", "z": "#1fa37a"}


class Camera:
    """Reads the webcam continuously so a frame is available instantly at any moment."""

    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise SystemExit("Cannot open webcam")
        self.frame = None
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.05)

    def snapshot(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


class Session:
    def __init__(self, out_dir, window_s, block, minutes, seed):
        self.window_s = window_s
        self.n_windows = int(minutes * 60 // window_s)
        self.win_dir = os.path.join(out_dir, "windows")
        self.strip_dir = os.path.join(out_dir, "strips")
        os.makedirs(self.win_dir, exist_ok=True)
        os.makedirs(self.strip_dir, exist_ok=True)

        # balanced cue schedule: rounds of [0,1,2] in random order, BLOCK windows each
        rng = random.Random(seed)
        self.schedule = []
        while len(self.schedule) < self.n_windows:
            order = [0, 1, 2]
            rng.shuffle(order)
            for a in order:
                self.schedule += [a] * block
        self.schedule = self.schedule[: self.n_windows]

        self.lock = threading.Lock()
        self.samples = []                 # (t, axis, value) of the current window
        self.latest = {}                  # axis -> (t, value) for row assembly
        self.rows = []                    # completed rows of the current window
        self.raw = open(os.path.join(out_dir, "raw_session.csv"), "a", newline="")
        self.raw_w = csv.writer(self.raw)
        log_path = os.path.join(out_dir, "session_log.csv")
        new_log = not os.path.exists(log_path)
        self.log = open(log_path, "a", newline="")
        self.log_w = csv.writer(self.log)
        if new_log:
            self.log_w.writerow(["filename", "seq", "window_start", "window_end", "rows", "cue"])
        self.seq_offset = len([f for f in os.listdir(self.win_dir) if f.endswith(".csv")])

        self.t0 = None
        self.seq = 0
        self.last = None                  # info about the last saved window for the dashboard
        self.done = False

    # called from the cloud thread for every value
    def on_sample(self, axis, value, t):
        with self.lock:
            self.raw_w.writerow([f"{t:.3f}", axis, value])
            self.latest[axis] = (t, float(value))
            if len(self.latest) == 3:     # all three axes updated -> one row
                t_row = max(v[0] for v in self.latest.values())
                self.rows.append((t_row, *(self.latest[a][1] for a in ("x", "y", "z"))))
                self.latest = {}

    def cue_at(self, k):
        return self.schedule[k] if 0 <= k < len(self.schedule) else None

    def run(self, camera, lead_s=0.0):
        self.t0 = time.time() + lead_s          # countdown shown on the dashboard before window 1
        time.sleep(max(0.0, self.t0 - time.time() - 3))
        if lead_s >= 3:
            beep()
        time.sleep(max(0.0, self.t0 - time.time()))
        with self.lock:                          # drop what arrived during the countdown
            self.rows, self.latest = [], {}
        for k in range(self.n_windows):
            w_start = self.t0 + k * self.window_s
            strip = []
            for frac in (0.1, 0.35, 0.6, 0.85):                # frames spread over the window
                time.sleep(max(0.0, w_start + frac * self.window_s - time.time()))
                strip.append(camera.snapshot())
                if frac == 0.6 and self.cue_at(k + 1) not in (None, self.cue_at(k)):
                    threading.Thread(target=beep_warn, daemon=True).start()   # 4 s before the change
            time.sleep(max(0.0, w_start + self.window_s - time.time()))
            w_end = time.time()
            with self.lock:
                rows, self.rows = self.rows, []
                self.raw.flush()
            frame = camera.snapshot()
            self.save_window(k, rows, frame, strip, w_start, w_end)
            if k + 1 < self.n_windows and self.cue_at(k + 1) != self.cue_at(k):
                threading.Thread(target=beep, daemon=True).start()
        self.done = True
        print("[capture] session finished")

    def save_window(self, k, rows, frame, strip, w_start, w_end):
        seq = self.seq_offset + k + 1
        stem = f"{seq}_{datetime.fromtimestamp(w_end).strftime('%Y%m%d%H%M%S')}"
        with open(os.path.join(self.win_dir, stem + ".csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "x", "y", "z"])
            for r in rows:
                w.writerow([f"{r[0]:.3f}", *r[1:]])
        if frame is not None:
            cv2.imwrite(os.path.join(self.win_dir, stem + ".jpg"), frame)
        frames = [s for s in strip if s is not None]
        if frames:
            small = [cv2.resize(s, (320, 240)) for s in frames]
            cv2.imwrite(os.path.join(self.strip_dir, stem + ".jpg"), cv2.hconcat(small))
        save_graph_png(rows, w_start, os.path.join(self.win_dir, stem + ".png"), stem)
        self.log_w.writerow([stem, seq, f"{w_start:.3f}", f"{w_end:.3f}", len(rows), self.cue_at(k)])
        self.log.flush()

        jpg = None
        if frame is not None:
            ok, buf = cv2.imencode(".jpg", cv2.resize(frame, (480, 360)))
            jpg = "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None
        with self.lock:
            self.seq = seq
            self.last = {"stem": stem, "rows": rows, "w_start": w_start, "img": jpg}
        print(f"[capture] {stem}: {len(rows)} rows, cue={self.cue_at(k)}")


def save_graph_png(rows, w_start, path, title):
    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    if rows:
        t = [r[0] - w_start for r in rows]
        for i, a in enumerate(("x", "y", "z")):
            ax.plot(t, [r[i + 1] for r in rows], "-o", ms=3, color=COLORS[a], label=a)
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("seconds in window")
    ax.set_ylabel("m/s²")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def beep():
    try:
        import winsound
        for _ in range(2):
            winsound.Beep(988, 120)
            time.sleep(0.08)
    except Exception:
        pass


def beep_warn():
    """Lower, single beep a few seconds before the activity changes."""
    try:
        import winsound
        winsound.Beep(523, 200)
    except Exception:
        pass


def build_app(session):
    app = dash.Dash(__name__, title="SIT225 6D capture")
    app.layout = html.Div(style={"fontFamily": "Segoe UI, sans-serif", "maxWidth": "1100px", "margin": "auto"}, children=[
        html.H2("Smartphone accelerometer + activity image capture"),
        html.Div(id="cue", style={"fontSize": "26px", "fontWeight": "bold", "padding": "12px",
                                  "borderRadius": "8px", "color": "white"}),
        html.Div(id="status", style={"margin": "8px 0", "color": "#444"}),
        html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
            dcc.Graph(id="graph", style={"flex": "1 1 560px", "height": "380px"}),
            html.Img(id="image", style={"width": "480px", "maxWidth": "100%", "border": "1px solid #ccc"}),
        ]),
        dcc.Interval(id="tick", interval=500),
    ])
    cue_color = {0: "#607d8b", 1: "#2a78d6", 2: "#e8543a", None: "#333"}
    shown = {"seq": -1}

    @app.callback(Output("cue", "children"), Output("cue", "style"), Output("status", "children"),
                  Output("graph", "figure"), Output("image", "src"), Input("tick", "n_intervals"))
    def refresh(_n):
        style = {"fontSize": "26px", "fontWeight": "bold", "padding": "12px", "borderRadius": "8px", "color": "white"}
        if session.t0 is None:
            style["background"] = "#333"
            return "Dang khoi dong ...", style, "", dash.no_update, dash.no_update
        now = time.time()
        if now < session.t0:                      # preparation countdown before window 1
            first = session.cue_at(0)
            style["background"] = cue_color[first]
            return (f"CHUAN BI - bat dau sau {session.t0 - now:.0f}s   |   hoat dong dau tien: {CUE_TEXT[first]}",
                    style, "Ngoi vao khung hinh, cam dien thoai san sang", dash.no_update, dash.no_update)
        k = int((now - session.t0) // session.window_s)
        cue = session.cue_at(k)
        remaining = session.window_s - (now - session.t0) % session.window_s
        nxt = session.cue_at(k + 1)
        style["background"] = cue_color[cue]
        if session.done or cue is None:
            text = "Xong roi - co the dung script (Ctrl+C)"
        else:
            text = f"{CUE_TEXT[cue]}   ({remaining:.0f}s)"
            if nxt is not None and nxt != cue:
                if remaining <= 5:                # highlight the upcoming change in its own colour
                    style["background"] = cue_color[nxt]
                    text = f"DOI SAU {remaining:.0f}s  ->  {CUE_TEXT[nxt]}"
                else:
                    text += f"   ->   sap toi: {SHORT[nxt]}"
        elapsed = min(now - session.t0, session.n_windows * session.window_s)
        status = (f"window {min(k + 1, session.n_windows)}/{session.n_windows} | elapsed {elapsed / 60:.1f} min | "
                  f"rows in current window: {len(session.rows)}")

        with session.lock:
            last = session.last
        if last is None or session.seq == shown["seq"]:
            return text, style, status, dash.no_update, dash.no_update
        shown["seq"] = session.seq
        fig = go.Figure()
        for i, a in enumerate(("x", "y", "z")):
            fig.add_scatter(x=[r[0] - last["w_start"] for r in last["rows"]], y=[r[i + 1] for r in last["rows"]],
                            mode="lines+markers", name=a, line={"color": COLORS[a]})
        fig.update_layout(title=f"{last['stem']}  ({len(last['rows'])} rows)", xaxis_title="seconds in window",
                          yaxis_title="m/s²", margin={"l": 50, "r": 10, "t": 40, "b": 40})
        return text, style, status, fig, last["img"] or dash.no_update

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=35)
    ap.add_argument("--window", type=float, default=10, help="seconds per window")
    ap.add_argument("--block", type=int, default=3, help="windows per cue block")
    ap.add_argument("--out", default="data")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--seed", type=int, default=225)
    ap.add_argument("--lead", type=float, default=45, help="preparation countdown before window 1 (seconds)")
    args = ap.parse_args()

    camera = Camera()
    session = Session(args.out, args.window, args.block, args.minutes, args.seed)
    start_phone_stream(session.on_sample)
    print("[capture] waiting 8 s for the cloud connection and camera ...")
    time.sleep(8)
    frame = camera.snapshot()
    if frame is None or frame.std() < 2:
        print("[capture] WARNING: webcam image is blank - open the privacy shutter / enable the camera")
    if not session.rows and not session.latest:
        print("[capture] WARNING: no accelerometer data yet - is IoT Remote open on the phone?")
    threading.Thread(target=session.run, args=(camera, args.lead), daemon=True).start()
    print(f"[capture] open http://127.0.0.1:{args.port}")
    build_app(session).run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
