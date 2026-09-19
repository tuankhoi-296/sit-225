"""
smooth_dash.py - smooth, real-time Plotly Dash graphs for any continuous data stream.

Usage (three lines are enough):

    from smooth_dash import smooth_live_dash

    live = smooth_live_dash(["x", "y", "z"], title="Accelerometer", y_title="m/s²")
    live.start()                 # Dash server runs in a background thread
    live.push("x", 0.12)         # call from any thread whenever a new value arrives

How it works (see the report for diagrams):

  producer threads --push()--> StreamBuffer (thread-safe ring buffer, sequence numbers)
        --every poll_ms, server callback sends ONLY samples newer than the client's cursor-->
  browser playout queue (jitter buffer, adaptive delay)
        --requestAnimationFrame render loop at `fps`-->
  one Plotly.react() per frame: append released points, interpolate the leading edge towards
  the next queued sample, drop expired ones, slide the time window and ease the y-axis range.

Because the x-axis scrolls with the clock (not with data arrival) and each sample is
revealed exactly when its timestamp reaches the right edge, the user sees a continuous
"tape" moving at constant speed, even when data arrives in bursts or at irregular rates.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import dash
from dash import Input, Output, State, dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots

__all__ = ["smooth_live_dash", "SmoothLiveDash", "StreamBuffer"]

_DEFAULT_COLORS = ["#2a78d6", "#e8543a", "#1fa37a", "#9a5bd1", "#e0a019", "#3fb5c9"]


class StreamBuffer:
    """Thread-safe ring buffer of timestamped samples.

    Every sample gets a monotonically increasing sequence number, so each browser tab
    only has to remember the last sequence number it received (its *cursor*) to ask
    for "everything newer than this". Old samples fall out automatically (deque maxlen).
    """

    def __init__(self, channels: Sequence[str], capacity: int = 20000):
        self.channels = list(channels)
        self._index = {name: i for i, name in enumerate(self.channels)}   # name -> trace index
        self._buf: deque = deque(maxlen=capacity)   # a full buffer drops its oldest sample by itself
        self._lock = threading.Lock()               # producers and the web server are different threads
        self._seq = 0                               # sequence number of the newest sample
        self.total_pushed = 0

    def push(self, channel: str, value: float, t: Optional[float] = None) -> None:
        """Add one sample. `t` is a Unix timestamp in seconds (default: now)."""
        if channel not in self._index:
            raise KeyError(f"Unknown channel {channel!r}; expected one of {self.channels}")

        try:
            value = float(value)
        except (TypeError, ValueError):
            return                                  # a sensor can send junk, so just skip it
        if math.isnan(value) or math.isinf(value):
            return                                  # a NaN would cut the line in Plotly

        t_ms = time.time() * 1000.0 if t is None else float(t) * 1000.0   # ms, the unit JavaScript uses

        with self._lock:
            self._seq += 1
            self.total_pushed += 1
            self._buf.append((self._seq, t_ms, self._index[channel], value))

    def since(self, cursor: Optional[int], not_before_ms: float, limit: int) -> Tuple[List[list], int, bool]:
        """Return (samples, new_cursor, initial).

        samples  : [[t_ms, channel_index, value], ...] in arrival order
        initial  : True when the client had no cursor (fresh page) - it receives recent
                   history (t >= not_before_ms) instead of only new samples.
        """
        with self._lock:
            last_seq = self._seq
            initial = cursor is None or cursor < 0 or cursor > last_seq   # fresh tab, or server restarted
            out = []

            # walk backwards, the newest samples sit at the end of the deque
            for seq, t_ms, ch, v in reversed(self._buf):
                if initial:
                    if t_ms < not_before_ms:
                        break
                elif seq <= cursor:
                    break
                out.append([int(t_ms), ch, round(v, 5)])   # rounding keeps the JSON small
                if len(out) >= limit:
                    break

        out.reverse()                                 # back to chronological order
        return out, last_seq, initial


# The renderer below runs in the browser, not in Python. Dash injects it as a clientside
# callback; it keeps its state in window.__smoothDash[graph_id] and redraws on every frame.
_CLIENTSIDE_JS = r"""
function(batch, cfg) {
    const dc = window.dash_clientside;
    const root = (window.__smoothDash = window.__smoothDash || {});
    let st = root[cfg.graph_id];

    if (!st) {
        const nCh = cfg.channels.length;
        st = root[cfg.graph_id] = {
            queue: [],                 // [t, ch, v] sorted by t, waiting to be revealed
            offset: null,              // serverClock - browserClock (ms)
            delay: cfg.delay_ms,       // current playout delay (adapts)
            lagPeak: 0,                // decaying max of observed sample lag
            lastT: new Array(nCh).fill(-Infinity),
            lastQT: new Array(nCh).fill(-Infinity),   // newest sample time received per channel
            gapEma: new Array(nCh).fill(null),        // typical interval between samples per channel
            ghost: new Array(nCh).fill(false),        // trace ends with an interpolated leading-edge point
            yr: cfg.y_axes.map(() => null),
            rev: 0, lastDraw: 0, lastStats: 0, paused: false,
            rxLog: [], frames: 0, fpsT0: performance.now(), fps: 0, late: 0
        };
        const plotDiv = () => document.querySelector('#' + CSS.escape(cfg.graph_id) + ' .js-plotly-plot');

        const ease = (cur, target, dt, tau) => cur + (target - cur) * (1 - Math.exp(-dt / tau));

        const draw = (nowPerf) => {
            const dt = Math.min(250, nowPerf - (st.lastDraw || nowPerf));
            if (nowPerf - st.lastDraw < 1000 / cfg.fps - 2) return;
            const gd = plotDiv();
            if (!gd || !gd._fullLayout || !window.Plotly || st.offset === null) return;
            st.lastDraw = nowPerf;
            if (st.paused) return;

            // 1. adapt the playout delay to how late samples actually arrive
            st.lagPeak *= Math.exp(-dt / 15000);       // forget an old spike after ~15 s
            const gapNeed = cfg.interpolate ? Math.max(0, ...st.gapEma.map(g => g || 0)) * 1.1 : 0;
            const target = Math.min(cfg.max_delay_ms,
                                    Math.max(cfg.delay_ms, st.lagPeak + gapNeed + cfg.margin_ms));
            st.delay = ease(st.delay, target, dt, target > st.delay ? 300 : 4000);   // up fast, down slowly

            const serverNow = Date.now() + st.offset;
            const playT = serverNow - st.delay;
            const left = playT - cfg.window_ms;

            // remove last frame's interpolated leading-edge points before adding real samples
            st.ghost.forEach((g, ch) => {
                if (g) { gd.data[ch].x.pop(); gd.data[ch].y.pop(); st.ghost[ch] = false; }
            });

            // 2. reveal samples whose timestamp has reached the right edge
            let n = 0;
            while (n < st.queue.length && st.queue[n][0] <= playT) n++;
            const released = n ? st.queue.splice(0, n) : [];
            for (const [t0, ch, v] of released) {
                const t = Math.max(t0, st.lastT[ch]);       // never draw backwards in time
                st.lastT[ch] = t;
                gd.data[ch].x.push(t);
                gd.data[ch].y.push(v);
            }

            // 2b. grow the line towards the next sample, which the queue already holds.
            //     Without this, 1 Hz data would appear one whole segment at a time.
            if (cfg.interpolate) {
                const seen = new Array(gd.data.length).fill(false);
                for (const [tq, ch, vq] of st.queue) {
                    if (seen[ch]) continue;
                    seen[ch] = true;
                    const tr = gd.data[ch], t1 = st.lastT[ch];
                    if (!tr.y.length || !(playT > t1) || !(tq > t1) || tq - t1 > cfg.max_gap_ms) continue;
                    const y1 = tr.y[tr.y.length - 1];
                    const f = Math.min(1, (playT - t1) / (tq - t1));   // how far between the two samples
                    tr.x.push(playT);
                    tr.y.push(y1 + f * (vq - y1));
                    st.ghost[ch] = true;                               // this point is removed next frame
                }
            }

            // 3. drop points that scrolled out (keep one just outside for a continuous line)
            const cutoff = left;
            for (const tr of gd.data) {
                let k = 0;
                while (k + 1 < tr.x.length && tr.x[k + 1] < cutoff) k++;
                const extra = Math.max(k, tr.x.length - cfg.max_points);
                if (extra > 0) { tr.x.splice(0, extra); tr.y.splice(0, extra); }
            }

            // 4. slide every x-axis and ease every y-axis towards the visible data range
            const lay = gd.layout;
            for (const ax of cfg.x_axes) {
                lay[ax] = lay[ax] || {};
                lay[ax].autorange = false;
                lay[ax].range = [left, playT];
            }
            cfg.y_axes.forEach((axCfg, i) => {
                const ax = axCfg.name;
                lay[ax] = lay[ax] || {};
                lay[ax].autorange = false;
                if (cfg.y_range) { lay[ax].range = cfg.y_range; return; }
                let lo = Infinity, hi = -Infinity;
                for (const idx of axCfg.traces) {
                    const tr = gd.data[idx];
                    for (let j = 0; j < tr.y.length; j++) {
                        if (tr.x[j] < cutoff) continue;
                        if (tr.y[j] < lo) lo = tr.y[j];
                        if (tr.y[j] > hi) hi = tr.y[j];
                    }
                }
                if (lo === Infinity) { if (!st.yr[i]) return; lo = st.yr[i][0]; hi = st.yr[i][1]; }
                const mid = (lo + hi) / 2, half = Math.max((hi - lo) / 2 * 1.15, cfg.min_y_span / 2);
                const tLo = mid - half, tHi = mid + half;
                if (!st.yr[i]) st.yr[i] = [tLo, tHi];
                // expand quickly so peaks are never clipped, contract gently so the axis does not jump
                st.yr[i][0] = ease(st.yr[i][0], tLo, dt, tLo < st.yr[i][0] ? 90 : 1500);
                st.yr[i][1] = ease(st.yr[i][1], tHi, dt, tHi > st.yr[i][1] ? 90 : 1500);
                lay[ax].range = [st.yr[i][0], st.yr[i][1]];
            });

            // 5. one redraw per frame
            lay.datarevision = ++st.rev;
            window.Plotly.react(gd, gd.data, lay);

            st.frames++;
            if (nowPerf - st.lastStats > 500) {
                const el = document.getElementById(cfg.stats_id);
                const since = nowPerf - st.fpsT0;
                st.fps = st.frames * 1000 / since; st.frames = 0; st.fpsT0 = nowPerf;
                const cutoffRx = Date.now() - 5000;
                while (st.rxLog.length && st.rxLog[0][0] < cutoffRx) st.rxLog.shift();
                const rate = st.rxLog.reduce((a, r) => a + r[1], 0) / 5;
                if (el) el.textContent =
                    `rx ${rate.toFixed(1)} samples/s  |  render ${st.fps.toFixed(0)} fps  |  ` +
                    `playout delay ${(st.delay / 1000).toFixed(2)} s  |  queued ${st.queue.length}  |  late ${st.late}`;
                st.lastStats = nowPerf;
            }
        };

        const loop = (ts) => {
            try { draw(ts); } catch (e) { console.error('smooth_dash render error', e); }
            window.requestAnimationFrame(loop);
        };
        window.requestAnimationFrame(loop);

        document.addEventListener('click', (e) => {
            if (!e.target || e.target.id !== cfg.pause_id) return;
            st.paused = !st.paused;
            e.target.textContent = st.paused ? 'Resume' : 'Pause';
        });
    }

    if (!batch || !batch.samples) return dc.no_update;

    // Latency can only make server_now look older than it is, so the largest value we ever
    // see is the best estimate of the real offset between the two clocks.
    const off = batch.server_now - Date.now();
    if (st.offset === null || off > st.offset) st.offset = off;
    else st.offset += (off - st.offset) * 0.02;

    const serverNow = Date.now() + st.offset;
    let count = 0;
    for (const s of batch.samples) {
        const gap = s[0] - st.lastQT[s[1]];
        if (gap > 0 && gap <= cfg.max_gap_ms) {
            const g = st.gapEma[s[1]];
            st.gapEma[s[1]] = g === null ? gap : g + (gap - g) * 0.1;
        }
        if (s[0] > st.lastQT[s[1]]) st.lastQT[s[1]] = s[0];
        if (!batch.initial) {
            const lag = serverNow - s[0];
            if (lag > st.lagPeak) st.lagPeak = lag;
            if (lag > st.delay) st.late++;
        }
        // sorted insert (samples are almost always already in order)
        let i = st.queue.length;
        while (i > 0 && st.queue[i - 1][0] > s[0]) i--;
        st.queue.splice(i, 0, s);
        count++;
    }
    if (st.queue.length > cfg.max_points * cfg.channels.length) {
        st.queue.splice(0, st.queue.length - cfg.max_points * cfg.channels.length);
    }
    if (!batch.initial && count) st.rxLog.push([Date.now(), count]);
    return dc.no_update;
}
"""


class SmoothLiveDash:
    """Handle returned by :func:`smooth_live_dash`. Push data in, run the server."""

    def __init__(
        self,
        channels: Sequence[str],
        *,
        title: str = "Live data",
        window_s: float = 20.0,
        delay_s: float = 0.6,
        max_delay_s: float = 5.0,
        fps: int = 30,
        poll_ms: int = 150,
        interpolate: bool = True,
        max_gap_s: float = 2.5,
        subplots: bool = True,
        y_title: Optional[str] = None,
        y_range: Optional[Tuple[float, float]] = None,
        min_y_span: float = 0.5,
        colors: Optional[Sequence[str]] = None,
        line_width: float = 2.0,
        height: Optional[int] = None,
        max_points: int = 3000,
        buffer_capacity: int = 50000,
        name: str = "live",
        app: Optional[dash.Dash] = None,
    ):
        if not channels:
            raise ValueError("channels must contain at least one name")
        self.channels = list(channels)
        self.buffer = StreamBuffer(self.channels, capacity=buffer_capacity)
        self.window_ms = float(window_s) * 1000
        self.max_delay_ms = float(max_delay_s) * 1000
        self.poll_ms = int(poll_ms)
        self._thread: Optional[threading.Thread] = None

        ids = {k: f"{name}-{k}" for k in ("graph", "poll", "cursor", "batch", "cfg", "sink", "stats", "pause")}
        self.ids = ids
        colors = list(colors) if colors else _DEFAULT_COLORS
        n = len(self.channels)

        # The figure starts out empty. Every point in it is added later by the browser.
        if subplots and n > 1:
            fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                subplot_titles=self.channels)
            for i, ch in enumerate(self.channels):
                fig.add_trace(self._trace(ch, colors[i % len(colors)], line_width), row=i + 1, col=1)
            x_axes = ["xaxis" if i == 0 else f"xaxis{i + 1}" for i in range(n)]
            y_axes = [{"name": "yaxis" if i == 0 else f"yaxis{i + 1}", "traces": [i]} for i in range(n)]
            default_height = 170 * n + 90
        else:
            fig = go.Figure([self._trace(ch, colors[i % len(colors)], line_width)
                             for i, ch in enumerate(self.channels)])
            x_axes = ["xaxis"]
            y_axes = [{"name": "yaxis", "traces": list(range(n))}]
            default_height = 460

        fig.update_xaxes(type="date", tickformat="%H:%M:%S", showgrid=True, gridcolor="#e6e8eb")
        fig.update_yaxes(showgrid=True, gridcolor="#e6e8eb", zeroline=False, title_text=y_title)
        fig.update_layout(
            height=height or default_height,
            margin=dict(l=60, r=20, t=40, b=40),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=not (subplots and n > 1),
            legend=dict(orientation="h", y=1.08, x=0),
            uirevision="smooth-dash",
            hovermode=False,
        )

        cfg = {
            "graph_id": ids["graph"], "stats_id": ids["stats"], "pause_id": ids["pause"],
            "channels": self.channels, "window_ms": self.window_ms,
            "delay_ms": float(delay_s) * 1000, "max_delay_ms": self.max_delay_ms,
            "margin_ms": 250.0, "fps": int(fps), "max_points": int(max_points),
            "interpolate": bool(interpolate), "max_gap_ms": float(max_gap_s) * 1000,
            "x_axes": x_axes, "y_axes": y_axes,
            "y_range": list(y_range) if y_range else None, "min_y_span": float(min_y_span),
        }

        self.app = app or dash.Dash(__name__, title=title, update_title=None)
        self.component = html.Div([
            html.Div([
                html.H3(title, style={"margin": "0", "fontWeight": "600"}),
                html.Button("Pause", id=ids["pause"], n_clicks=0,
                            style={"marginLeft": "auto", "padding": "4px 14px", "cursor": "pointer"}),
            ], style={"display": "flex", "alignItems": "center", "gap": "12px"}),
            html.Div("connecting…", id=ids["stats"],
                     style={"fontFamily": "Consolas, monospace", "fontSize": "12px", "color": "#5b6470",
                            "margin": "6px 0 4px"}),
            dcc.Graph(id=ids["graph"], figure=fig, config={"displayModeBar": False},
                      style={"height": f"{height or default_height}px"}),
            dcc.Interval(id=ids["poll"], interval=self.poll_ms),   # ticks a few times per second
            dcc.Store(id=ids["cursor"], data=None),   # last sequence number this tab has received
            dcc.Store(id=ids["batch"], data=None),    # the new samples, handed over to JavaScript
            dcc.Store(id=ids["cfg"], data=cfg),       # settings the render loop reads
            html.Div(id=ids["sink"], style={"display": "none"}),   # clientside callbacks need an output
        ], style={"fontFamily": "Segoe UI, Roboto, Helvetica, Arial, sans-serif", "padding": "12px 16px"})
        if app is None:
            self.app.layout = self.component

        max_batch = int(max_points) * n
        history_ms = self.window_ms + self.max_delay_ms + 2000   # how far back a new tab gets filled

        @self.app.callback(
            Output(ids["batch"], "data"),
            Output(ids["cursor"], "data"),
            Input(ids["poll"], "n_intervals"),
            State(ids["cursor"], "data"),
        )
        def _poll(_n, cursor):
            """Send this tab the samples it has not seen yet - never the whole figure."""
            now_ms = time.time() * 1000.0
            samples, new_cursor, initial = self.buffer.since(cursor, now_ms - history_ms, max_batch)

            if not samples and not initial:
                return dash.no_update, dash.no_update   # nothing new, so leave the browser alone

            # server_now lets the browser line its own clock up with ours
            return {"server_now": now_ms, "samples": samples, "initial": initial}, new_cursor

        self.app.clientside_callback(
            _CLIENTSIDE_JS,
            Output(ids["sink"], "children"),
            Input(ids["batch"], "data"),
            State(ids["cfg"], "data"),
        )

    @staticmethod
    def _trace(name: str, color: str, width: float) -> go.Scatter:
        return go.Scatter(x=[], y=[], name=name, mode="lines",
                          line=dict(color=color, width=width), hoverinfo="skip")

    # ---- the methods a user of this module actually calls ----
    def push(self, channel: str, value: float, t: Optional[float] = None) -> None:
        """Add one value to `channel`. Thread-safe. `t` = Unix time in seconds (default now)."""
        self.buffer.push(channel, value, t)

    def push_row(self, values: Dict[str, float], t: Optional[float] = None) -> None:
        """Add several channels sampled at the same instant, e.g. {"x": .1, "y": .2, "z": 9.8}."""
        t = time.time() if t is None else t
        for ch, v in values.items():
            self.buffer.push(ch, v, t)

    def run(self, host: str = "127.0.0.1", port: int = 8050, **kwargs) -> None:
        """Start the Dash server and block (use from the main thread)."""
        kwargs.setdefault("debug", False)
        logging.getLogger("werkzeug").setLevel(logging.WARNING)   # polling would spam the console
        self.app.run(host=host, port=port, **kwargs)

    def start(self, host: str = "127.0.0.1", port: int = 8050, **kwargs) -> threading.Thread:
        """Start the Dash server in a daemon thread and return immediately."""
        if self._thread and self._thread.is_alive():
            return self._thread
        kwargs["debug"] = False
        kwargs["use_reloader"] = False       # the reloader would start the data source twice
        self._thread = threading.Thread(target=self.run, args=(host, port), kwargs=kwargs,
                                        name="smooth-dash-server", daemon=True)
        self._thread.start()
        print(f"[smooth_dash] dashboard running at http://{host}:{port}")
        return self._thread


def smooth_live_dash(channels: Iterable[str], **options) -> SmoothLiveDash:
    """Create a smooth real-time Dash dashboard for continuous data.

    Parameters
    ----------
    channels    : names of the signals, one line per channel (e.g. ["x", "y", "z"]).
    title       : page heading.
    window_s    : seconds of history visible on screen (default 20).
    delay_s     : minimum playout delay; data is shown this far behind real time so bursts
                  can be smoothed out. Grows automatically (up to max_delay_s) when data
                  arrives later than this (default 0.6).
    fps         : target redraw rate in the browser (default 30).
    poll_ms     : how often the browser asks the server for new samples (default 150).
    interpolate : grow each line smoothly towards the next sample (default True). The playout
                  delay then also adapts to the typical sample interval, so low-rate data
                  (e.g. 1 Hz from Arduino IoT Cloud) is drawn continuously.
    max_gap_s   : no interpolation across gaps longer than this (default 2.5 s).
    subplots    : True = one stacked subplot per channel, False = all lines on one plot.
    y_title, y_range, min_y_span, colors, line_width, height : appearance.
    app         : pass an existing dash.Dash app to embed `handle.component` in your own layout.

    Returns
    -------
    SmoothLiveDash with .push(channel, value, t=None), .push_row(dict, t=None),
    .start(host, port) (non-blocking), .run(host, port) (blocking), .app, .component
    """
    return SmoothLiveDash(list(channels), **options)
