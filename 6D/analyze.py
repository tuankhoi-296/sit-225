"""
analyze.py - compute window features, compare activities and make the report figures.
"""
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

DATA, FIG = "data", "figures"
NAMES = {0: "no activity", 1: "waving", 2: "shaking"}
CLS_COLORS = {0: "#607d8b", 1: "#2a78d6", 2: "#e8543a"}
AX_COLORS = {"x": "#2a78d6", "y": "#e8543a", "z": "#1fa37a"}
MIN_ROWS = 3
FEATURES = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z", "mag_mean", "mag_std", "mag_range", "mag_jerk"]

# Thresholds of the simple rule, picked by looking at the windows labelled confidently from the images.
FLAT_JERK = 1.52
UPRIGHT_Y = 1.51
WAVE_STD_Z = 3.50

os.makedirs(FIG, exist_ok=True)


def read_window(stem):
    return pd.read_csv(os.path.join(DATA, "windows", stem + ".csv"))


def features(df):
    f = {"rows": len(df)}
    if len(df) == 0:
        return f
    mag = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)
    for a in "xyz":
        f[f"mean_{a}"] = df[a].mean()
        f[f"std_{a}"] = df[a].std(ddof=0)
    f["mag_mean"] = mag.mean()
    f["mag_std"] = mag.std(ddof=0)
    f["mag_range"] = mag.max() - mag.min()
    f["mag_jerk"] = np.abs(np.diff(mag)).mean() if len(mag) > 1 else 0.0
    return f


def simple_rule(row):
    # Flat data: the phone is lying on the desk.
    if row.mag_jerk <= FLAT_JERK:
        return 0

    # Gravity is not on the y axis: the phone is held flat, so it is being shaken.
    if row.mean_y <= UPRIGHT_Y:
        return 2

    # Phone upright: a wide swing on z means waving, otherwise shaking.
    return 1 if row.std_z > WAVE_STD_Z else 2


def plot_window(ax, stem, title=None):
    df = read_window(stem)
    if len(df):
        t = df.timestamp - df.timestamp.iloc[0]
        for a in "xyz":
            ax.plot(t, df[a], "-o", ms=2.5, lw=1.2, color=AX_COLORS[a], label=a)
    ax.set_ylim(-25, 25)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)
    ax.set_title(title or stem, fontsize=8)


def show_image(ax, stem, strip=False):
    path = os.path.join(DATA, "strips" if strip else "windows", stem + ".jpg")
    img = cv2.imread(path)
    if img is not None:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis("off")


def main():
    ann = pd.read_csv(os.path.join(DATA, "annotations.csv"))
    notes_path = os.path.join(DATA, "annotation_notes.csv")
    notes = pd.read_csv(notes_path) if os.path.exists(notes_path) else ann.assign(uncertain=0)
    log = pd.read_csv(os.path.join(DATA, "session_log.csv"))

    rows = []
    for stem in ann.filename:
        rows.append({"filename": stem, **features(read_window(stem))})
    feat = pd.DataFrame(rows).merge(ann, on="filename").merge(notes[["filename", "uncertain"]], on="filename", how="left")
    feat = feat.merge(log[["filename", "seq", "cue"]], on="filename", how="left")
    feat.to_csv(os.path.join(DATA, "features.csv"), index=False)

    # Window 1 also holds the samples buffered during the 45 s countdown, so it is excluded.
    # (capture_dash.py was fixed afterwards to clear that buffer.)
    usable = feat[(feat.rows >= MIN_ROWS) & (feat.seq > 1)].copy()

    res = {
        "windows_total": int(len(feat)),
        "windows_usable": int(len(usable)),
        "windows_too_few_rows": int((feat.rows < MIN_ROWS).sum()),
        "rows_per_window_mean": float(feat.rows.mean()),
        "rows_per_window_min": int(feat.rows.min()),
        "rows_per_window_max": int(feat.rows.max()),
        "label_counts": {NAMES[k]: int(v) for k, v in feat.label.value_counts().sort_index().items()},
        "uncertain": int(feat.uncertain.fillna(0).sum()),
    }

    # ---- image labels vs cue schedule ----

    # All comparisons use the usable windows only.
    both = usable.dropna(subset=["cue"])
    cm_cue = confusion_matrix(both.cue.astype(int), both.label, labels=[0, 1, 2])
    res["annotation_vs_cue_agreement"] = float((both.cue.astype(int) == both.label).mean())
    res["annotation_vs_cue_cm"] = cm_cue.tolist()
    disagree = both[both.cue.astype(int) != both.label]
    res["annotation_vs_cue_disagreements"] = disagree[["filename", "label", "cue", "rows", "uncertain"]].to_dict("records")

    # ---- per-class statistics ----

    stats = usable.groupby("label")[FEATURES].agg(["mean", "std"])
    res["class_stats"] = {
        NAMES[k]: {f: [round(float(stats.loc[k, (f, "mean")]), 3), round(float(stats.loc[k, (f, "std")]), 3)]
                   for f in FEATURES}
        for k in stats.index
    }

    # ---- step 7: re-decide the uncertain windows from the data ----

    usable["rule"] = usable.apply(simple_rule, axis=1)
    unc_rows = usable[usable.uncertain.fillna(0) == 1]
    if len(unc_rows):
        new_labels = unc_rows.rule.values
        final = usable.set_index("filename").label.copy()
        final.loc[unc_rows.filename] = new_labels
        res["resolved_changed"] = int((new_labels != unc_rows.label.values).sum())
        res["resolved_total"] = int(len(unc_rows))
        res["final_vs_cue_agreement"] = float((final.loc[usable.filename].values == usable.cue.astype(int).values).mean())
        res["final_vs_cue_cm"] = confusion_matrix(usable.cue.astype(int), final.loc[usable.filename], labels=[0, 1, 2]).tolist()
        res["final_label_counts"] = {NAMES[k]: int(v) for k, v in final.value_counts().sort_index().items()}

        out = final.reset_index()
        out.columns = ["filename", "label"]
        out = out.assign(seq=out.filename.str.split("_").str[0].astype(int)).sort_values("seq")
        out[["filename", "label"]].to_csv(os.path.join(DATA, "annotations_final.csv"), index=False)

    # ---- figures ----

    rng = np.random.default_rng(1)

    # Examples: 3 instances per class (graph + image).
    fig, axes = plt.subplots(6, 3, figsize=(11, 15), gridspec_kw={"height_ratios": [1, 1.1] * 3})
    examples = {}
    for c in range(3):
        pool = usable[(usable.label == c) & (usable.uncertain.fillna(0) == 0)]
        pick = pool.sample(min(3, len(pool)), random_state=int(rng.integers(1e6))).sort_values("seq") if len(pool) else pool
        examples[NAMES[c]] = list(pick.filename)
        for j in range(3):
            ax_g, ax_i = axes[2 * c, j], axes[2 * c + 1, j]
            if j < len(pick):
                stem = pick.filename.iloc[j]
                plot_window(ax_g, stem, f"{NAMES[c]}: {stem}")
                show_image(ax_i, stem)
            else:
                ax_g.axis("off")
                ax_i.axis("off")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_examples.png"), dpi=110)
    plt.close(fig)
    res["examples"] = examples

    # Overlay of all windows per class (each axis and the magnitude).
    fig, axes = plt.subplots(3, 4, figsize=(13, 8), sharex=True, sharey="col")
    for c in range(3):
        for stem in usable[usable.label == c].filename:
            df = read_window(stem)
            t = df.timestamp - df.timestamp.iloc[0]
            mag = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)
            for j, s in enumerate([df.x, df.y, df.z, mag]):
                axes[c, j].plot(t, s, color=CLS_COLORS[c], alpha=0.25, lw=0.9)
        for j, name in enumerate(["x", "y", "z", "|a| magnitude"]):
            axes[0, j].set_title(name)
            axes[c, j].grid(alpha=0.3)
        axes[c, 0].set_ylabel(f"{NAMES[c]}\nm/s²")
    for j in range(4):
        axes[2, j].set_xlabel("seconds in window")
    fig.suptitle("All windows overlaid, per annotated class")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_overlay.png"), dpi=110)
    plt.close(fig)

    # Boxplots of the main features.
    key = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z", "mag_std", "mag_jerk"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for ax, f in zip(axes.flat, key):
        data = [usable[usable.label == c][f] for c in range(3)]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], range(3)):
            patch.set_facecolor(CLS_COLORS[c])
            patch.set_alpha(0.6)
        ax.set_xticks([1, 2, 3], [NAMES[c] for c in range(3)], fontsize=8)
        ax.set_title(f, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_boxplots.png"), dpi=110)
    plt.close(fig)

    # Scatter coloured by the activity that was actually cued.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for c in range(3):
        d = usable[usable.cue == c]
        axes[0].scatter(d.mean_y, d.mag_jerk, s=22, color=CLS_COLORS[c], label=NAMES[c], alpha=0.85)
        axes[1].scatter(d.mean_z, d.mag_std, s=22, color=CLS_COLORS[c], label=NAMES[c], alpha=0.85)
    axes[0].set_xlabel("mean y (m/s²) - phone upright?")
    axes[0].set_ylabel("mean |Δ|a|| (m/s²) - movement intensity")
    axes[1].set_xlabel("mean z (m/s²)")
    axes[1].set_ylabel("std of |a| (m/s²)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Same features, coloured by the activity the participant was cued to perform")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_scatter_cue.png"), dpi=110)
    plt.close(fig)

    # ---- hard cases ----

    # A window is hard if it was marked uncertain, disagrees with the cue,
    # or its image label disagrees with what the simple rule reads from the data.
    hard = set(feat[feat.uncertain.fillna(0) == 1].filename)
    hard |= set(disagree.filename)
    hard |= set(usable[usable.rule != usable.label].filename)
    hard = feat[feat.filename.isin(hard)].sort_values("seq")
    cols = ["filename", "label", "cue", "rows", "uncertain"] + (["mag_std", "mean_z", "mean_y"] if len(hard) else [])
    res["difficult"] = hard[cols].to_dict("records")

    show = hard.head(6)
    if len(show):
        fig, axes = plt.subplots(len(show), 2, figsize=(12, 2.6 * len(show)), gridspec_kw={"width_ratios": [2.2, 1]})
        axes = np.atleast_2d(axes)
        for r, (_, row) in enumerate(show.iterrows()):
            show_image(axes[r, 0], row.filename, strip=True)
            cue = "?" if pd.isna(row.cue) else NAMES[int(row.cue)]
            uncertain = ", marked uncertain" if row.uncertain == 1 else ""
            axes[r, 0].set_title(f"{row.filename}: annotated {NAMES[row.label]}, cue {cue}{uncertain}", fontsize=9)
            plot_window(axes[r, 1], row.filename, f"{row.rows} rows, std|a|={row.get('mag_std', float('nan')):.2f}")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig_difficult.png"), dpi=100)
        plt.close(fig)
    res["difficult_shown"] = list(show.filename)

    with open(os.path.join(DATA, "results.json"), "w") as f:
        json.dump(res, f, indent=2,
                  default=lambda o: None if pd.isna(o) else (int(o) if isinstance(o, np.integer) else float(o)))

    skip = ("difficult", "annotation_vs_cue_disagreements", "class_stats")
    print(json.dumps({k: v for k, v in res.items() if k not in skip}, indent=2, default=str))


if __name__ == "__main__":
    main()
