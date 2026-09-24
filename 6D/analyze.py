"""
analyze.py - pattern analysis of the annotated accelerometer windows.

Inputs : data/windows/*.csv, data/annotations.csv, data/annotation_notes.csv, data/session_log.csv
Outputs: data/features.csv, data/results.json, figures/*.png
"""
import csv
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, export_text

DATA, FIG = "data", "figures"
NAMES = {0: "no activity", 1: "waving", 2: "shaking"}
CLS_COLORS = {0: "#607d8b", 1: "#2a78d6", 2: "#e8543a"}
AX_COLORS = {"x": "#2a78d6", "y": "#e8543a", "z": "#1fa37a"}
MIN_ROWS = 3
FEATURES = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z", "mag_mean", "mag_std", "mag_range", "mag_jerk"]
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


def cohen_d(a, b):
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return 0.0 if sp == 0 or np.isnan(sp) else (a.mean() - b.mean()) / sp


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
    # window 1 of this session also holds the samples buffered during the 45 s preparation countdown
    # (fixed in capture_dash.py afterwards), so its 10 s of data are not comparable - exclude it.
    usable = feat[(feat.rows >= MIN_ROWS) & (feat.seq > 1)].copy()
    res = {"windows_total": int(len(feat)), "windows_usable": int(len(usable)),
           "windows_too_few_rows": int((feat.rows < MIN_ROWS).sum()),
           "rows_per_window_mean": float(feat.rows.mean()), "rows_per_window_min": int(feat.rows.min()),
           "rows_per_window_max": int(feat.rows.max()),
           "label_counts": {NAMES[k]: int(v) for k, v in feat.label.value_counts().sort_index().items()},
           "uncertain": int(feat.uncertain.fillna(0).sum())}

    # ---- annotation (image) vs cue schedule --------------------------------------------------------
    both = usable.dropna(subset=["cue"])   # all comparisons use the usable windows only
    cm_cue = confusion_matrix(both.cue.astype(int), both.label, labels=[0, 1, 2])
    res["annotation_vs_cue_agreement"] = float((both.cue.astype(int) == both.label).mean())
    res["annotation_vs_cue_cm"] = cm_cue.tolist()
    disagree = both[both.cue.astype(int) != both.label]
    res["annotation_vs_cue_disagreements"] = disagree[["filename", "label", "cue", "rows", "uncertain"]].to_dict("records")

    # ---- per-class statistics ---------------------------------------------------------------------
    stats = usable.groupby("label")[FEATURES].agg(["mean", "std"])
    res["class_stats"] = {NAMES[k]: {f: [round(float(stats.loc[k, (f, "mean")]), 3), round(float(stats.loc[k, (f, "std")]), 3)]
                                     for f in FEATURES} for k in stats.index}
    contrasts = {}
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        A, B = usable[usable.label == a], usable[usable.label == b]
        if len(A) > 1 and len(B) > 1:
            d = {f: round(float(cohen_d(A[f], B[f])), 2) for f in FEATURES}
            contrasts[f"{NAMES[a]} vs {NAMES[b]}"] = dict(sorted(d.items(), key=lambda kv: -abs(kv[1])))
    res["contrasts_cohen_d"] = contrasts

    # ---- interpretable classifier: labels from images, cross-validated -----------------------------
    X, y = usable[FEATURES].values, usable.label.values
    n_splits = int(min(5, np.bincount(y).min())) if len(set(y)) == 3 else 0
    if n_splits >= 2:
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=0)
        pred = cross_val_predict(tree, X, y, cv=StratifiedKFold(n_splits, shuffle=True, random_state=0))
        usable["pred"] = pred
        res["tree_cv_accuracy"] = float((pred == y).mean())
        res["tree_cv_cm"] = confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()
        tree.fit(X, y)
        res["tree_rules"] = export_text(tree, feature_names=FEATURES, decimals=2)
        res["tree_mismatches"] = usable[usable.pred != usable.label][["filename", "label", "pred", "cue", "rows"]].to_dict("records")
        imp = dict(zip(FEATURES, tree.feature_importances_))
        res["tree_importance"] = {k: round(float(v), 3) for k, v in sorted(imp.items(), key=lambda kv: -kv[1]) if v > 0}

    # ---- how well does the DATA separate the protocol classes? (upper bound for the sensor) ----
    if n_splits >= 2 and usable.cue.notna().all():
        y_cue = usable.cue.astype(int).values
        tree_c = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=0)
        pred_c = cross_val_predict(tree_c, X, y_cue, cv=StratifiedKFold(n_splits, shuffle=True, random_state=0))
        res["tree_cv_accuracy_vs_cue"] = float((pred_c == y_cue).mean())
        res["tree_cv_cm_vs_cue"] = confusion_matrix(y_cue, pred_c, labels=[0, 1, 2]).tolist()
        tree_c.fit(X, y_cue)
        res["tree_rules_vs_cue"] = export_text(tree_c, feature_names=FEATURES, decimals=2)

    # ---- step 7: resolve the windows the annotator was unsure about, using the data ----
    # A tree is trained ONLY on the windows labelled confidently from the images, then used to
    # re-decide the uncertain ones. This is the "analyse the data to settle ambiguous images" step.
    conf = usable[usable.uncertain.fillna(0) == 0]
    unc_rows = usable[usable.uncertain.fillna(0) == 1]
    if len(unc_rows) and conf.label.nunique() == 3:
        resolver = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=0).fit(conf[FEATURES], conf.label)
        new_labels = resolver.predict(unc_rows[FEATURES])
        final = usable.set_index("filename").label.copy()
        final.loc[unc_rows.filename] = new_labels
        res["resolved_changed"] = int((new_labels != unc_rows.label.values).sum())
        res["resolved_total"] = int(len(unc_rows))
        res["final_vs_cue_agreement"] = float((final.loc[usable.filename].values == usable.cue.astype(int).values).mean())
        res["final_vs_cue_cm"] = confusion_matrix(usable.cue.astype(int), final.loc[usable.filename], labels=[0, 1, 2]).tolist()
        res["final_label_counts"] = {NAMES[k]: int(v) for k, v in final.value_counts().sort_index().items()}
        out = final.reset_index()
        out.columns = ["filename", "label"]
        out.assign(seq=out.filename.str.split("_").str[0].astype(int)).sort_values("seq")[["filename", "label"]] \
            .to_csv(os.path.join(DATA, "annotations_final.csv"), index=False)
        res["resolver_rules"] = export_text(resolver, feature_names=FEATURES, decimals=2)

    # ---- figures ----------------------------------------------------------------------------------
    rng = np.random.default_rng(1)
    # examples: 3 instances per class (graph + image)
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
                ax_g.axis("off"); ax_i.axis("off")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_examples.png"), dpi=110)
    plt.close(fig)
    res["examples"] = examples

    # overlay of all windows per class (magnitude and each axis)
    fig, axes = plt.subplots(3, 4, figsize=(13, 8), sharex=True, sharey="col")
    for c in range(3):
        for stem in usable[usable.label == c].filename:
            df = read_window(stem)
            t = df.timestamp - df.timestamp.iloc[0]
            mag = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)
            for j, (name, s) in enumerate([("x", df.x), ("y", df.y), ("z", df.z), ("|a|", mag)]):
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

    # boxplots of key features
    key = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z", "mag_std", "mag_jerk"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for ax, f in zip(axes.flat, key):
        data = [usable[usable.label == c][f] for c in range(3)]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], range(3)):
            patch.set_facecolor(CLS_COLORS[c]); patch.set_alpha(0.6)
        ax.set_xticks([1, 2, 3], [NAMES[c] for c in range(3)], fontsize=8)
        ax.set_title(f, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_boxplots.png"), dpi=110)
    plt.close(fig)

    # 2-D feature scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for c in range(3):
        d = usable[usable.label == c]
        axes[0].scatter(d.mean_z, d.mag_std, s=22, color=CLS_COLORS[c], label=NAMES[c], alpha=0.8)
        axes[1].scatter(d.mean_y, d.std_z, s=22, color=CLS_COLORS[c], label=NAMES[c], alpha=0.8)
    axes[0].set_xlabel("mean z (m/s²) - phone orientation"); axes[0].set_ylabel("std of |a| (m/s²) - movement intensity")
    axes[1].set_xlabel("mean y (m/s²)"); axes[1].set_ylabel("std z (m/s²)")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend()
    unc = usable[usable.uncertain.fillna(0) == 1]
    axes[0].scatter(unc.mean_z, unc.mag_std, s=90, facecolors="none", edgecolors="k", label="uncertain")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_scatter.png"), dpi=110)
    plt.close(fig)

    # same feature space, coloured by the protocol (cue) class instead of the image label
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for c in range(3):
        d = usable[usable.cue == c]
        axes[0].scatter(d.mean_y, d.mag_jerk, s=22, color=CLS_COLORS[c], label=NAMES[c], alpha=0.85)
        axes[1].scatter(d.mean_z, d.mag_std, s=22, color=CLS_COLORS[c], label=NAMES[c], alpha=0.85)
    axes[0].set_xlabel("mean y (m/s²) - phone upright?"); axes[0].set_ylabel("mean |Δ|a|| (m/s²) - movement intensity")
    axes[1].set_xlabel("mean z (m/s²)"); axes[1].set_ylabel("std of |a| (m/s²)")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Same features, coloured by the activity the participant was cued to perform")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_scatter_cue.png"), dpi=110)
    plt.close(fig)

    # confusion matrices
    n_mats = 2 + ("tree_cv_cm_vs_cue" in res)
    fig, axes = plt.subplots(1, n_mats, figsize=(5 * n_mats, 4.2))
    mats = [("Image annotation vs cue schedule", cm_cue, "annotation (image)", "cue")]
    if "tree_cv_cm" in res:
        mats.append(("Decision tree (5-fold CV) vs annotation", np.array(res["tree_cv_cm"]), "predicted from data", "annotation"))
    if "tree_cv_cm_vs_cue" in res:
        mats.append(("Decision tree (5-fold CV) vs cue", np.array(res["tree_cv_cm_vs_cue"]), "predicted from data", "cue"))
    for ax, (title, m, xl, yl) in zip(axes, mats):
        ax.imshow(m, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, int(m[i, j]), ha="center", va="center", color="white" if m[i, j] > m.max() / 2 else "black")
        ax.set_xticks(range(3), [NAMES[c] for c in range(3)]); ax.set_yticks(range(3), [NAMES[c] for c in range(3)])
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_confusion.png"), dpi=110)
    plt.close(fig)

    # difficult cases: uncertain, or annotation disagrees with cue, or tree disagrees with annotation
    hard = set(feat[feat.uncertain.fillna(0) == 1].filename) | set(disagree.filename)
    if "pred" in usable:
        hard |= set(usable[usable.pred != usable.label].filename)
    hard = feat[feat.filename.isin(hard)].sort_values("seq")
    res["difficult"] = hard[["filename", "label", "cue", "rows", "uncertain"] + (["mag_std", "mean_z", "mean_y"] if len(hard) else [])].to_dict("records")
    show = hard.head(6)
    if len(show):
        fig, axes = plt.subplots(len(show), 2, figsize=(12, 2.6 * len(show)), gridspec_kw={"width_ratios": [2.2, 1]})
        axes = np.atleast_2d(axes)
        for r, (_, row) in enumerate(show.iterrows()):
            show_image(axes[r, 0], row.filename, strip=True)
            cue = "?" if pd.isna(row.cue) else NAMES[int(row.cue)]
            axes[r, 0].set_title(f"{row.filename}: annotated {NAMES[row.label]}, cue {cue}"
                                 f"{', marked uncertain' if row.uncertain == 1 else ''}", fontsize=9)
            plot_window(axes[r, 1], row.filename, f"{row.rows} rows, std|a|={row.get('mag_std', float('nan')):.2f}")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig_difficult.png"), dpi=100)
        plt.close(fig)
    res["difficult_shown"] = list(show.filename)

    with open(os.path.join(DATA, "results.json"), "w") as f:
        json.dump(res, f, indent=2, default=lambda o: None if pd.isna(o) else (int(o) if isinstance(o, np.integer) else float(o)))
    print(json.dumps({k: v for k, v in res.items() if k not in ("tree_rules", "difficult", "tree_mismatches",
                                                                "annotation_vs_cue_disagreements", "class_stats")}, indent=2, default=str))
    print(res.get("tree_rules", ""))


if __name__ == "__main__":
    main()
