import argparse
import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _read_run_csv(path: str) -> Tuple[List[float], List[float]]:
    nwt_by_ep: Dict[int, float] = {}
    vqs_by_ep: Dict[int, float] = {}
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ep = int(row["episode"])
            nwt_by_ep[ep] = float(row["nwt"])
            vqs_by_ep[ep] = float(row["vqs"])
    max_ep = max(nwt_by_ep.keys()) if nwt_by_ep else -1
    nwt = [nwt_by_ep[ep] for ep in range(max_ep + 1)]
    vqs = [vqs_by_ep[ep] for ep in range(max_ep + 1)]
    return nwt, vqs


def _plot_series(values: List[float], out_png: str, title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    if not values:
        return
    episodes = list(range(len(values)))
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(episodes, values, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("episode")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot avg_nwt and avg_vqs from run CSVs.")
    p.add_argument("--training-dir", default=os.path.join("results", "training"))
    p.add_argument("--runs", type=int, nargs="+", default=[1, 2, 3])
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    training_dir = os.path.abspath(args.training_dir)
    runs = [int(r) for r in args.runs]

    all_nwt: List[List[float]] = []
    all_vqs: List[List[float]] = []

    for run_idx in runs:
        path = os.path.join(training_dir, f"run{run_idx}.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing run CSV: {path}")
        nwt, vqs = _read_run_csv(path)
        all_nwt.append(nwt)
        all_vqs.append(vqs)

    if not all_nwt or not all_vqs:
        raise RuntimeError("No run data loaded.")

    episodes = min(len(r) for r in all_nwt)
    avg_nwt = [float(np.mean([run[ep] for run in all_nwt])) for ep in range(episodes)]
    avg_vqs = [float(np.mean([run[ep] for run in all_vqs])) for ep in range(episodes)]

    out_nwt = os.path.join(training_dir, "avg_nwt.png")
    out_vqs = os.path.join(training_dir, "avg_vqs.png")
    _plot_series(avg_nwt, out_nwt, "Avg cumulative negative wait time", "nwt")
    _plot_series(avg_vqs, out_vqs, "Avg cumulative vehicle queue size", "vqs")
    print(f"Saved: {out_nwt}")
    print(f"Saved: {out_vqs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
