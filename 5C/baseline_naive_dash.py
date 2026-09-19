"""
baseline_naive_dash.py - the Week 8 approach, kept for comparison.

Waits until N new samples have arrived, then rebuilds and re-sends the WHOLE figure.
The user sees the graph freeze, then jump. Compare with accel_live_dash.py.

Run:  python baseline_naive_dash.py [--replay data/x.csv]   then open http://127.0.0.1:8052
"""
import argparse
import threading

import dash
import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from plotly.subplots import make_subplots

from phone_stream import start_csv_replay, start_phone_stream

N = 30                                  # samples to collect before the figure is rebuilt
lock = threading.Lock()
pending = {"x": [], "y": [], "z": []}   # arrived but not shown yet
shown = {"x": [], "y": [], "z": []}     # the block currently on screen


def on_sample(axis, value, t):
    with lock:
        pending[axis].append(value)     # the timestamp is thrown away, the x-axis is just an index


app = dash.Dash(__name__, title="Naive Dash (Week 8 baseline)")
app.layout = html.Div([
    html.H3(f"Naive update: redraw the whole figure every {N} samples"),
    dcc.Graph(id="g"),
    dcc.Interval(id="tick", interval=500),
], style={"fontFamily": "Segoe UI, Arial, sans-serif", "padding": "12px 16px"})


@app.callback(Output("g", "figure"), Input("tick", "n_intervals"))
def refresh(_n):
    with lock:
        if len(pending["z"]) < N:
            return dash.no_update       # this is the freeze: nothing moves until N samples are in

        for a in shown:
            shown[a] = pending[a][:N]    # and this is the jump: a completely new block of data
            pending[a] = pending[a][N:]

    # A whole new figure is built and sent to the browser every time, axes included.
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["x", "y", "z"])
    for i, a in enumerate(["x", "y", "z"]):
        fig.add_trace(go.Scatter(y=shown[a], mode="lines", name=a), row=i + 1, col=1)

    fig.update_layout(height=600, showlegend=False)
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay")
    args = ap.parse_args()
    if args.replay:
        start_csv_replay(args.replay, on_sample)
    else:
        start_phone_stream(on_sample)

    app.run(port=8052, debug=False)      # a different port, so it can run next to the smooth one
