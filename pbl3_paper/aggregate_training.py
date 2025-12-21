import argparse
import csv
import os
from typing import Dict, List, Optional, Sequence

import numpy as np


def read_csv(path: str) -> List[Dict[str, object]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_avg(rows: List[Dict[str, object]], out_png: str) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    episodes = [int(r["episode"]) for r in rows]
    nwt_mean = [float(r["sum_neg_reward_mean"]) for r in rows]
    vqs_mean = [float(r["sum_intersection_queue_mean"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(episodes, nwt_mean, color="tab:blue", linewidth=1.5)
    ax1.set_title("Cumulative negative wait times across episodes (avg)")
    ax1.set_xlabel("episode")
    ax1.set_ylabel("cumulative negative wait time")
    ax1.grid(True, alpha=0.3)

    ax2.plot(episodes, vqs_mean, color="tab:purple", linewidth=1.5)
    ax2.set_title("Cumulative intersection queue size across episodes (avg)")
    ax2.set_xlabel("episode")
    ax2.set_ylabel("cumulative intersection queue size")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate multiple train_log.csv runs (paper-style average).")
    p.add_argument("--logs", nargs="+", required=True, help="Paths to train_log.csv files")
    p.add_argument("--outdir", required=True, help="Output directory")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    runs = [read_csv(p) for p in args.logs]
    by_ep: Dict[int, Dict[str, List[float]]] = {}

    for run in runs:
        for row in run:
            try:
                ep = int(row["episode"])
            except Exception:
                continue
            rec = by_ep.setdefault(ep, {"sum_neg_reward": [], "sum_intersection_queue": []})
            if "sum_neg_reward" in row:
                rec["sum_neg_reward"].append(float(row["sum_neg_reward"]))
            if "sum_intersection_queue" in row:
                rec["sum_intersection_queue"].append(float(row["sum_intersection_queue"]))

    out_rows: List[Dict[str, object]] = []
    for ep in sorted(by_ep.keys()):
        rec = by_ep[ep]
        nwt = np.array(rec["sum_neg_reward"], dtype=np.float64)
        vqs = np.array(rec["sum_intersection_queue"], dtype=np.float64)
        out_rows.append(
            {
                "episode": int(ep),
                "n_runs": int(max(len(nwt), len(vqs))),
                "sum_neg_reward_mean": float(np.mean(nwt)) if nwt.size else 0.0,
                "sum_neg_reward_std": float(np.std(nwt)) if nwt.size else 0.0,
                "sum_intersection_queue_mean": float(np.mean(vqs)) if vqs.size else 0.0,
                "sum_intersection_queue_std": float(np.std(vqs)) if vqs.size else 0.0,
            }
        )

    out_csv = os.path.join(args.outdir, "train_log_avg.csv")
    out_png = os.path.join(args.outdir, "training_paper_avg.png")
    write_csv(out_csv, out_rows)
    plot_avg(out_rows, out_png)
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
