# validate.py
"""
Run the full pipeline on the validation set and report:
  • precision / recall / F1 for 'made' vs everything else
  • avg latency per clip
  • writes logs/metrics.csv
"""

import csv, json, time, statistics, pathlib, sys
from typing import Dict, List

# ----- import your high-level entrypoint -----------------------------
from processor.gemini_processor import extract_events_from_video

VAL_DIR   = pathlib.Path("tests/val")
CLIPS_DIR = VAL_DIR / "clips"
GT_CSV    = VAL_DIR / "ground_truth.csv"
METRIC_CSV= pathlib.Path("logs/metrics.csv")
METRIC_CSV.parent.mkdir(exist_ok=True, parents=True)

def load_ground_truth() -> Dict[str, str]:
    with open(GT_CSV, newline="") as f:
        rdr = csv.DictReader(f)
        return {row["clip"]: row["result"] for row in rdr}

def evaluate(pred: List[dict]) -> str:
    """Return 'made', 'missed', or 'other' for the clip."""
    if not pred:
        return "other"
    # majority vote if multiple events
    made_cnt  = sum(1 for e in pred if e["result"] == "made")
    missed_cnt= len(pred) - made_cnt
    return "made" if made_cnt >= missed_cnt else "missed"

def main() -> None:
    gt = load_ground_truth()
    y_true, y_pred, latencies = [], [], []

    with open(METRIC_CSV, "w", newline="") as outf:
        wr = csv.writer(outf)
        wr.writerow(["clip", "truth", "pred", "latency_s", "raw_events"])

        for clip_path in sorted(CLIPS_DIR.glob("*.mp4")):
            clip_name = clip_path.name
            truth     = gt.get(clip_name, "other")

            t0 = time.time()
            events = extract_events_from_video(clip_path)
            latency = time.time() - t0

            pred = evaluate(events)

            wr.writerow([clip_name, truth, pred,
                         f"{latency:.2f}", json.dumps(events)])
            y_true.append(truth)
            y_pred.append(pred)
            latencies.append(latency)
            print(f"{clip_name:>12}  truth={truth:<6}  pred={pred:<6}  "
                  f"{latency:5.2f}s")

    # ----- metrics ---------------------------------------------------
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "made" and p == "made")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != "made" and p == "made")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "made" and p != "made")

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    avg_lat   = statistics.mean(latencies) if latencies else 0

    print("\n=== Validation summary ===")
    print(f"Precision : {precision:5.2%}")
    print(f"Recall    : {recall:5.2%}")
    print(f"F1 score  : {f1:5.2%}")
    print(f"Avg time  : {avg_lat:5.2f} s per clip")
    print(f"Metrics logged to {METRIC_CSV}")

if __name__ == "__main__":
    if not VAL_DIR.exists():
        sys.exit("❌  tests/val directory not found.")
    main()
