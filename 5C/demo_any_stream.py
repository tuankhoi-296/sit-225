"""
demo_any_stream.py - smooth_live_dash with a stream that has nothing to do with the phone.

Two producers run at the same time to show the function is data-agnostic:
  * "cpu %"   : real CPU usage from psutil, sampled every 0.25 s
  * "signal"  : a 5 Hz synthetic wave delivered in BURSTS once per second (with the
                original sample timestamps) - the worst case for a naive dashboard.

Run:  python demo_any_stream.py   then open http://127.0.0.1:8051
"""
import math
import random
import threading
import time

import psutil

from smooth_dash import smooth_live_dash


def cpu_producer(live):
    """A steady 4 Hz source with no timestamps of its own."""
    while True:
        live.push("cpu %", psutil.cpu_percent(interval=None))   # push() stamps it with the time now
        time.sleep(0.25)


def bursty_producer(live, rate_hz=5.0):
    """Samples at 5 Hz but delivers a whole second at once, like a slow network would."""
    pending = []
    t0 = time.time()
    next_flush = t0 + 1.0

    while True:
        t = time.time()
        v = 2.0 * math.sin(2 * math.pi * 0.2 * (t - t0)) + 0.6 * math.sin(2 * math.pi * 1.1 * (t - t0)) \
            + random.gauss(0, 0.08)              # two sine waves plus a bit of noise
        pending.append((t, v))

        if t >= next_flush:
            for ts, val in pending:
                live.push("signal", val, t=ts)   # pass the real sample time, not the arrival time
            pending.clear()
            next_flush += 1.0

        time.sleep(1.0 / rate_hz)


if __name__ == "__main__":
    # Same wrapper as the accelerometer dashboard, only the channel names differ.
    live = smooth_live_dash(["cpu %", "signal"], title="smooth_live_dash demo - any continuous data",
                            window_s=15, delay_s=0.5)

    threading.Thread(target=cpu_producer, args=(live,), daemon=True).start()
    threading.Thread(target=bursty_producer, args=(live,), daemon=True).start()

    live.run(port=8051)   # the delay grows to about 1.1 s by itself because of the bursts
