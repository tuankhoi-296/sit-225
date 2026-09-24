"""
annotate.py - label every captured window from its webcam images only.

Shows the 4-frame strip (data/strips) above the end-of-window image (data/windows).
The cue schedule is deliberately NOT shown, so labels come from the images alone.

Keys:  0 = no activity   1 = waving   2 = shaking
       u = mark current window as "uncertain" (then press 0/1/2)
       b = back one window   q = save and quit (re-run to resume)

Output:
    data/annotations.csv        filename,label        (the 2-column annotation file)
    data/annotation_notes.csv   filename,label,uncertain
"""
import csv
import os

import cv2
import numpy as np

DATA = "data"
NAMES = {0: "no activity", 1: "waving", 2: "shaking"}


def load(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {r["filename"]: r for r in csv.DictReader(f)}


def save(stems, labels, uncertain):
    with open(os.path.join(DATA, "annotations.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        for s in stems:
            if s in labels:
                w.writerow([s, labels[s]])
    with open(os.path.join(DATA, "annotation_notes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "uncertain"])
        for s in stems:
            if s in labels:
                w.writerow([s, labels[s], int(s in uncertain)])


def main():
    win_dir = os.path.join(DATA, "windows")
    stems = sorted({f[:-4] for f in os.listdir(win_dir) if f.endswith(".csv")}, key=lambda s: int(s.split("_")[0]))
    notes = load(os.path.join(DATA, "annotation_notes.csv"))
    labels = {s: int(r["label"]) for s, r in notes.items()}
    uncertain = {s for s, r in notes.items() if r.get("uncertain") == "1"}
    i = next((k for k, s in enumerate(stems) if s not in labels), len(stems) - 1)
    flag = False

    while 0 <= i < len(stems):
        s = stems[i]
        main_img = cv2.imread(os.path.join(win_dir, s + ".jpg"))
        strip = cv2.imread(os.path.join(DATA, "strips", s + ".jpg"))
        main_img = cv2.resize(main_img, (640, 480)) if main_img is not None else np.zeros((480, 640, 3), np.uint8)
        if strip is not None:
            strip = cv2.resize(strip, (640, int(strip.shape[0] * 640 / strip.shape[1])))
            canvas = cv2.vconcat([strip, main_img])
        else:
            canvas = main_img
        bar = np.full((60, canvas.shape[1], 3), 30, np.uint8)
        cur = f"current: {NAMES[labels[s]]}" if s in labels else "unlabelled"
        cv2.putText(bar, f"{i + 1}/{len(stems)}  {s}  [{cur}]{'  UNCERTAIN' if flag else ''}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(bar, "0 none  1 waving  2 shaking  u uncertain  b back  q quit", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1)
        cv2.imshow("annotate", cv2.vconcat([bar, canvas]))
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("0"), ord("1"), ord("2")):
            labels[s] = key - ord("0")
            (uncertain.add if flag else uncertain.discard)(s)
            flag = False
            save(stems, labels, uncertain)
            i += 1
        elif key == ord("u"):
            flag = not flag
        elif key == ord("b"):
            i, flag = max(0, i - 1), False
        elif key in (ord("q"), 27):
            break
    save(stems, labels, uncertain)
    cv2.destroyAllWindows()
    print(f"labelled {len(labels)}/{len(stems)} windows ({len(uncertain)} uncertain)")


if __name__ == "__main__":
    main()
