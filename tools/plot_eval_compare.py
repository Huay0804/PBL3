import argparse
import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _gaussian_kde(xs: np.ndarray, samples: np.ndarray) -> np.ndarray:
    if samples.size <= 1:
        return np.zeros_like(xs)
    std = float(np.std(samples, ddof=1))
    if std <= 0.0:
        return np.zeros_like(xs)
    n = float(samples.size)
    bandwidth = 1.06 * std * (n ** (-1.0 / 5.0))
    if bandwidth <= 0.0:
        return np.zeros_like(xs)
    diffs = (xs[:, None] - samples[None, :]) / bandwidth
    kernel = np.exp(-0.5 * diffs * diffs)
    coef = 1.0 / (bandwidth * np.sqrt(2.0 * np.pi))
    return coef * np.mean(kernel, axis=1)


def _read_eval_csv(path: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {
        "fds_nwt_abs": [],
        "fds_vqs": [],
        "adap_nwt_abs": [],
        "adap_vqs": [],
    }
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out["fds_nwt_abs"].append(float(row["fds_nwt_abs"]))
            out["fds_vqs"].append(float(row["fds_vqs"]))
            out["adap_nwt_abs"].append(float(row["adap_nwt_abs"]))
            out["adap_vqs"].append(float(row["adap_vqs"]))
    return out


def _stats(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def _plot_panel(ax, series: List[Tuple[str, List[float], str]], title: str) -> None:
    if not series:
        return
    all_vals = [v for _, vals, _ in series for v in vals]
    if not all_vals:
        return
    xmin = float(min(all_vals))
    xmax = float(max(all_vals))
    xs = np.linspace(xmin, xmax, 300)
    for label, vals, color in series:
        arr = np.array(vals, dtype=np.float64)
        ax.hist(arr, bins=15, density=True, alpha=0.35, color=color, label=label)
        ax.plot(xs, _gaussian_kde(xs, arr), color=color, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel(title)
    ax.set_ylabel("density")
    ax.grid(False)
    ax.legend()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot FDS vs DQN baseline vs DDQN (hist + KDE).")
    p.add_argument("--baseline", required=True, help="Path to eval.csv from baseline DQN")
    p.add_argument("--ddqn", required=True, help="Path to eval.csv from DDQN+Dueling")
    p.add_argument("--outdir", default=os.path.join("results", "eval"))
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    base_csv = os.path.abspath(args.baseline)
    ddqn_csv = os.path.abspath(args.ddqn)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    base = _read_eval_csv(base_csv)
    ddqn = _read_eval_csv(ddqn_csv)

    # Prefer FDS from baseline; warn if mean differs a lot across files.
    fds_nwt_base = base["fds_nwt_abs"]
    fds_nwt_ddqn = ddqn["fds_nwt_abs"]
    fds_vqs_base = base["fds_vqs"]
    fds_vqs_ddqn = ddqn["fds_vqs"]

    if fds_nwt_ddqn and abs(np.mean(fds_nwt_base) - np.mean(fds_nwt_ddqn)) > 1e-3:
        print("WARN: FDS nwt_abs mean differs between baseline and ddqn eval files.")
    if fds_vqs_ddqn and abs(np.mean(fds_vqs_base) - np.mean(fds_vqs_ddqn)) > 1e-3:
        print("WARN: FDS vqs mean differs between baseline and ddqn eval files.")

    series_nwt = [
        ("FDS TLCS", fds_nwt_base, "#F4A261"),
        ("DQN Baseline", base["adap_nwt_abs"], "#4C78A8"),
        ("DDQN+Dueling", ddqn["adap_nwt_abs"], "#8CD17D"),
    ]
    series_vqs = [
        ("FDS TLCS", fds_vqs_base, "#F4A261"),
        ("DQN Baseline", base["adap_vqs"], "#4C78A8"),
        ("DDQN+Dueling", ddqn["adap_vqs"], "#8CD17D"),
    ]

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _plot_panel(axes[0], series_nwt, "Cumulative Negative Wait Time")
    _plot_panel(axes[1], series_vqs, "Cumulative Vehicle Queue Size")
    fig.tight_layout()
    out_png = os.path.join(outdir, "compare_hist.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    stats_path = os.path.join(outdir, "compare_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as handle:
        for label, values, _ in series_nwt:
            mean, std = _stats(values)
            handle.write(f"{label} nwt_abs mean={mean:.3f} std={std:.3f}\n")
        for label, values, _ in series_vqs:
            mean, std = _stats(values)
            handle.write(f"{label} vqs mean={mean:.3f} std={std:.3f}\n")

    print(f"Saved: {out_png}")
    print(f"Saved: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
