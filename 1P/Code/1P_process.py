"""
SIT225 Pass Task 1 - Data Analysis Pipeline
============================================
Purpose: From two raw CSV files (PIR + DHT22), reconstruct the PIR signal at
one-second resolution, apply ground-truth labels from the activity script,
compute the false-negative rate, and generate the two plots for Part D.

Run:
    python 1P_analysis.py

Required input files (place in the same folder):
    - pir_raw.csv   : columns [pc_datetime, arduino_ms, state]  (state = HIGH/LOW)
    - dht_raw.csv   : columns [pc_datetime, arduino_ms, humidity, temperature]

Outputs:
    - 1P_pir_with_groundtruth.csv   : per-second PIR signal + ground-truth column
    - 1P_summary_by_phase.csv       : summary table per activity phase
    - fig1_timeseries.png           : Figure 1 (PIR vs actual presence)
    - fig2_bar.png                  : Figure 2 (% HIGH per activity)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-GUI backend (works on any machine)
import matplotlib.pyplot as plt


# =====================================================================
# CONFIG - EDIT HERE IF NEEDED
# =====================================================================
PIR_FILE = "pir_log.csv"
DHT_FILE = "dht22_log.csv"

# The activity script you followed during data capture (minute markers : activity).
# THIS IS the ground truth - it is your own record of what you were actually doing.
# present = 1 (present), 0 (absent), None (ignore - warm-up).
# Edit to match your own capture session.
SCRIPT = [
    # (start_minute, end_minute, present, activity_name)
    (0,  1,  None, "warm-up"),
    (1,  6,  1,    "still: reading"),
    (6,  9,  1,    "moving"),
    (9,  15, 1,    "still: focused"),
    (15, 18, 0,    "away"),
    (18, 24, 1,    "normal work"),
    (24, 99, 1,    "still: final"),   # 99 = from minute 24 to end of session
]

WARMUP_SEC = 60                # seconds to drop at the start (PIR not yet stable)


# =====================================================================
# STEP 1 - LOAD RAW DATA
# =====================================================================
def load_raw():
    """Read both raw CSV files. Keep only the needed columns."""
    pir = pd.read_csv(PIR_FILE)
    dht = pd.read_csv(DHT_FILE)

    # Normalise column names to lowercase, strip whitespace
    pir.columns = [c.strip().lower() for c in pir.columns]
    dht.columns = [c.strip().lower() for c in dht.columns]

    # arduino_ms is the only reliable clock (pc_datetime wraps every 60 minutes)
    pir = pir[["arduino_ms", "state"]].copy()
    pir["arduino_ms"] = pir["arduino_ms"].astype(int)
    pir["state"] = pir["state"].str.strip().str.upper()   # "HIGH"/"LOW"

    dht = dht[["arduino_ms", "humidity", "temperature"]].copy()
    dht["arduino_ms"] = dht["arduino_ms"].astype(int)
    return pir, dht


# =====================================================================
# STEP 2 - RECONSTRUCT THE PIR SIGNAL AT ONE-SECOND RESOLUTION (forward-fill)
# =====================================================================
def reconstruct_pir(pir, t0, t_end):
    """
    The PIR logs by event (each row = one state change), so it is irregular.
    We rebuild it into a regular 1-second signal by 'forward-filling':
    at each second, hold the state of the most recent event before it.
    Assumption: before the first HIGH event, the state = LOW.
    """
    timeline = np.arange(t0, t_end, 1000)         # markers every 1000ms = 1 second
    events = pir[["arduino_ms", "state"]].values.tolist()

    states = []
    cur = "LOW"       # initial assumption
    idx = 0
    for t in timeline:
        # update state if we have reached an event time
        while idx < len(events) and events[idx][0] <= t:
            cur = events[idx][1]
            idx += 1
        states.append(1 if cur == "HIGH" else 0)

    tl = pd.DataFrame({"arduino_ms": timeline, "pir": states})
    tl["minute"] = (tl["arduino_ms"] - t0) / 60000.0   # convert ms -> minutes since t0
    return tl


# =====================================================================
# STEP 3 - APPLY GROUND-TRUTH LABELS FROM THE SCRIPT
# =====================================================================
def label_ground_truth(minute):
    """
    Return (present, activity_name) for a given minute marker,
    based on the SCRIPT above.
    This is the step that turns your 'record' into data.
    """
    for start, end, present, name in SCRIPT:
        if start <= minute < end:
            return present, name
    return None, "end"


def apply_ground_truth(tl):
    """Apply ground-truth labels across the whole timeline."""
    labels = tl["minute"].apply(label_ground_truth)
    tl["present"] = [x[0] for x in labels]
    tl["activity"] = [x[1] for x in labels]
    return tl


# =====================================================================
# STEP 4 - COMPUTE METRICS (false negative, confusion matrix...)
# =====================================================================
def compute_metrics(tl):
    """Compare the 'pir' column (sensor guess) with 'present' (ground truth)."""
    # Only consider seconds that have a ground-truth label (drop warm-up = None)
    valid = tl.dropna(subset=["present"]).copy()
    valid["present"] = valid["present"].astype(int)

    # Confusion matrix
    tp = int(((valid["present"] == 1) & (valid["pir"] == 1)).sum())  # present & PIR says occupied
    fn = int(((valid["present"] == 1) & (valid["pir"] == 0)).sum())  # present & PIR says empty (MISSED)
    fp = int(((valid["present"] == 0) & (valid["pir"] == 1)).sum())  # absent & PIR says occupied (false alarm)
    tn = int(((valid["present"] == 0) & (valid["pir"] == 0)).sum())  # absent & PIR says empty (correct)

    present_sec = tp + fn
    fn_rate = fn / present_sec * 100 if present_sec else 0
    accuracy = (tp + tn) / (tp + fn + fp + tn) * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) else 0

    print("=" * 55)
    print("CONFUSION MATRIX (per second)")
    print("=" * 55)
    print(f"                  PIR=HIGH   PIR=LOW")
    print(f"Actual: PRESENT     {tp:5d}     {fn:5d}  <- FN (missed)")
    print(f"Actual: ABSENT      {fp:5d}     {tn:5d}")
    print("-" * 55)
    print(f"False Negative Rate : {fn_rate:.1f}%   (headline figure)")
    print(f"Accuracy            : {accuracy:.1f}%")
    print(f"Precision           : {precision:.1f}%")
    print(f"Recall              : {recall:.1f}%")
    print(f"Time present        : {present_sec} s = {present_sec/60:.1f} min")
    print(f"Of which PIR missed : {fn} s = {fn/60:.1f} min")
    return valid


# =====================================================================
# STEP 5 - SUMMARY TABLE BY ACTIVITY
# =====================================================================
def summarise_by_phase(tl):
    rows = []
    for start, end, present, name in SCRIPT:
        if present is None:
            continue
        sub = tl[tl["activity"] == name]
        if len(sub) == 0:
            continue
        high_pct = sub["pir"].mean() * 100
        rows.append({
            "phase": name,
            "actual": "Present" if present == 1 else "Absent",
            "seconds": len(sub),
            "pir_high_pct": round(high_pct, 1),
            "pir_missed_pct": round(100 - high_pct, 1) if present == 1 else "",
        })
    df = pd.DataFrame(rows)
    df.to_csv("1P_summary_by_phase.csv", index=False)
    print("\n" + "=" * 55)
    print("SUMMARY BY ACTIVITY")
    print("=" * 55)
    print(df.to_string(index=False))
    return df


# =====================================================================
# STEP 6 - PLOT THE TWO FIGURES
# =====================================================================
def plot_timeseries(tl):
    """Figure 1: PIR vs actual presence over time, background shaded by activity."""
    phase_colors = {
        "warm-up": "#e0e0e0", "still: reading": "#ffcccc", "moving": "#cce5ff",
        "still: focused": "#ffcccc", "away": "#d4edda",
        "normal work": "#cce5ff", "still: final": "#ffcccc",
    }
    fig, ax = plt.subplots(figsize=(12, 4.5))

    # Shade each phase
    prev, start = None, None
    for _, r in tl.iterrows():
        if r["activity"] != prev:
            if prev in phase_colors:
                ax.axvspan(start, r["minute"], color=phase_colors[prev], alpha=0.5, zorder=0)
            prev, start = r["activity"], r["minute"]
    if prev in phase_colors:
        ax.axvspan(start, tl["minute"].max(), color=phase_colors[prev], alpha=0.5, zorder=0)

    ax.step(tl["minute"], tl["pir"], where="post", color="#c0392b",
            linewidth=1.4, label="PIR state", zorder=3)
    # ground truth: only plot labelled points
    gt = tl.dropna(subset=["present"])
    ax.step(gt["minute"], gt["present"].astype(float), where="post", color="#27ae60",
            linewidth=1.8, linestyle="--", label="Actual presence (ground truth)", zorder=2)

    ax.set_ylim(-0.15, 1.25)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["LOW / Absent", "HIGH / Present"])
    ax.set_xlabel("Time (minutes)")
    ax.set_title("Figure 1: PIR State vs Actual Presence over Time\n"
                 "Red = sitting still | Blue = moving | Green = away")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout(); plt.savefig("fig1_timeseries.png", dpi=130); plt.close()
    print("\n[saved fig1_timeseries.png]")


def plot_bar(tl):
    """Figure 2: % PIR HIGH per activity."""
    phases = [s for s in SCRIPT if s[2] is not None]
    names = [s[3] for s in phases]
    vals, cols = [], []
    for start, end, present, name in phases:
        sub = tl[tl["activity"] == name]
        vals.append(sub["pir"].mean() * 100)
        if present == 0:
            cols.append("#3498db")                     # absent = blue
        elif "still" in name:
            cols.append("#e74c3c")                     # sitting still = red
        else:
            cols.append("#f39c12")                     # moving/working = orange

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(names)), vals, color=cols, edgecolor="black", linewidth=0.7)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontweight="bold")
        lbl = "ABSENT" if phases[i][2] == 0 else "PRESENT"
        ax.text(i, -6, lbl, ha="center", fontsize=8, color="gray")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(": ", ":\n") for n in names], fontsize=9)
    ax.set_ylabel("% of time PIR reads HIGH ('occupied')")
    ax.set_ylim(-9, 105)
    ax.axhline(50, color="gray", linestyle=":", alpha=0.5)
    ax.set_title("Figure 2: Proportion of PIR 'occupied' readings by activity")
    plt.tight_layout(); plt.savefig("fig2_bar.png", dpi=130); plt.close()
    print("[saved fig2_bar.png]")


# =====================================================================
# STEP 7 - DHT22 DESCRIPTIVE STATISTICS (contextual)
# =====================================================================
def describe_dht(dht):
    print("\n" + "=" * 55)
    print("DHT22 DESCRIPTIVE STATISTICS (contextual)")
    print("=" * 55)
    print(f"Temperature : {dht['temperature'].min():.1f} - {dht['temperature'].max():.1f} C "
          f"(mean {dht['temperature'].mean():.1f}, std {dht['temperature'].std():.2f})")
    print(f"Humidity    : {dht['humidity'].min():.1f} - {dht['humidity'].max():.1f} % "
          f"(mean {dht['humidity'].mean():.1f}, std {dht['humidity'].std():.2f})")


# =====================================================================
# MAIN
# =====================================================================
def main():
    pir, dht = load_raw()

    # Time origin: minute 0 = first logged sample across both streams.
    # NOTE: if the board was running before you sat down, adjust t0 accordingly.
    t0 = min(pir["arduino_ms"].min(), dht["arduino_ms"].min())
    t_end = max(pir["arduino_ms"].max(), dht["arduino_ms"].max())
    print(f"Session: {(t_end - t0)/60000:.1f} min  (t0={t0}ms, t_end={t_end}ms)\n")

    # Pipeline
    tl = reconstruct_pir(pir, t0, t_end)
    tl = apply_ground_truth(tl)

    # Drop the warm-up at the start
    tl = tl[tl["arduino_ms"] >= t0 + WARMUP_SEC * 1000].copy()

    compute_metrics(tl)
    summarise_by_phase(tl)
    describe_dht(dht)
    plot_timeseries(tl)
    plot_bar(tl)

    # Export the CSV with the ground-truth column
    out = tl[["arduino_ms", "minute", "pir", "present", "activity"]].copy()
    out.columns = ["arduino_ms", "minute", "pir_state", "ground_truth_present", "activity"]
    out.to_csv("1P_pir_with_groundtruth.csv", index=False)
    print("\n[saved 1P_pir_with_groundtruth.csv]")
    print("\nDONE.")


if __name__ == "__main__":
    main()
