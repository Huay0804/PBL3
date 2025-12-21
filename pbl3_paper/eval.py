import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence

import numpy as np
import tensorflow as tf

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PBL3_ROOT = os.path.normpath(os.path.join(THIS_DIR, ".."))
for p in (THIS_DIR, PBL3_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from baseline_controllers import FixedTimePolicy, MaxQueuePolicy, run_episode  # noqa: E402
from env_sumo_tl import SumoTLEnv  # noqa: E402


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


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


@dataclass
class DQNPolicy:
    model: tf.keras.Model

    def reset(self) -> None:
        return

    def act(self, obs: np.ndarray, _info: dict) -> int:
        q = self.model.predict(obs[None, :], verbose=0)[0]
        return int(np.argmax(q))


def summarize(rows: List[Dict[str, object]], *, out_csv: str) -> None:
    import math

    modes = sorted(set(str(r["mode"]) for r in rows))
    # Paper-style metrics:
    # - nwt_abs: abs(sum of negative rewards per episode)
    # - sum_intersection_queue: sum(queue) sampled at each decision step
    metrics = ["nwt_abs", "sum_intersection_queue", "throughput_junction"]
    out: List[Dict[str, object]] = []
    for mode in modes:
        sub = [r for r in rows if str(r["mode"]) == mode]
        row: Dict[str, object] = {"mode": mode, "n": len(sub)}
        for m in metrics:
            vals = [float(r.get(m, 0.0)) for r in sub]
            mean = float(np.mean(vals)) if vals else 0.0
            std = float(np.std(vals)) if vals else 0.0
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_sem"] = std / math.sqrt(len(vals)) if len(vals) > 0 else 0.0
        out.append(row)
    write_csv(out_csv, out)


def _normal_pdf(xs: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std <= 0.0:
        return np.zeros_like(xs)
    z = (xs - float(mean)) / float(std)
    return (1.0 / (float(std) * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z * z)


def plot_eval(rows: List[Dict[str, object]], out_png: str) -> None:
    import matplotlib.pyplot as plt

    modes = sorted(set(str(r["mode"]) for r in rows))
    # If we have a learned policy, mirror the paper plots (Fixed vs Adaptive).
    plot_modes = ["fixed", "dqn"] if ("fixed" in modes and "dqn" in modes) else modes

    metrics = [
        ("nwt_abs", "Cumulative Negative Wait Time (abs)"),
        ("sum_intersection_queue", "Cumulative Vehicle Queue Size"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (key, title) in zip(axes, metrics):
        series = {}
        for mode in plot_modes:
            vals = np.array([float(r.get(key, 0.0)) for r in rows if str(r["mode"]) == mode], dtype=np.float64)
            series[mode] = vals

        all_vals = np.concatenate([v for v in series.values() if v.size > 0]) if series else np.array([], dtype=np.float64)
        if all_vals.size == 0:
            continue

        xmin = float(np.min(all_vals))
        xmax = float(np.max(all_vals))
        if xmin == xmax:
            xmin -= 1.0
            xmax += 1.0
        xs = np.linspace(xmin, xmax, 300)

        for mode in plot_modes:
            vals = series.get(mode, np.array([], dtype=np.float64))
            if vals.size == 0:
                continue
            ax.hist(vals, bins=15, density=True, alpha=0.35, label=mode)
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            if std > 0.0:
                ax.plot(xs, _normal_pdf(xs, mean, std), linewidth=1.6)

        ax.set_title(title)
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_paired_diffs(rows: List[Dict[str, object]], out_png: str) -> None:
    import matplotlib.pyplot as plt

    def collect(metric: str, a: str, b: str) -> np.ndarray:
        by_seed: Dict[int, Dict[str, Dict[str, object]]] = {}
        for r in rows:
            try:
                seed = int(r["seed"])
            except Exception:
                continue
            by_seed.setdefault(seed, {})[str(r.get("mode"))] = r
        diffs: List[float] = []
        for seed in sorted(by_seed.keys()):
            row_a = by_seed[seed].get(a)
            row_b = by_seed[seed].get(b)
            if row_a is None or row_b is None:
                continue
            diffs.append(float(row_b.get(metric, 0.0)) - float(row_a.get(metric, 0.0)))
        return np.array(diffs, dtype=np.float64)

    diffs_nwt = collect("nwt_abs", "fixed", "dqn")
    diffs_vqs = collect("sum_intersection_queue", "fixed", "dqn")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, diffs, title in [
        (axes[0], diffs_nwt, "Cumulative Negative Wait Time (Adaptive - Fixed)"),
        (axes[1], diffs_vqs, "Cumulative Vehicle Queue Size (Adaptive - Fixed)"),
    ]:
        if diffs.size == 0:
            continue
        ax.hist(diffs, bins=15, density=True, alpha=0.6, color="tab:blue")
        mean = float(np.mean(diffs))
        std = float(np.std(diffs)) if diffs.size > 1 else 0.0
        ax.set_title(title)
        ax.axvline(mean, color="black", linestyle="--", linewidth=1.2, label=f"mean={mean:.2f}")
        if std > 0.0:
            xs = np.linspace(float(np.min(diffs)), float(np.max(diffs)), 200)
            ax.plot(xs, _normal_pdf(xs, mean, std), color="tab:orange", linewidth=1.4)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def paired_left_ttest(rows: List[Dict[str, object]], *, metric: str, a: str = "fixed", b: str = "dqn") -> Dict[str, float]:
    """
    Paired difference test (b - a) with left-tailed alternative mean_diff < 0.
    Returns an approximate p-value using Normal(0,1) CDF (no SciPy).
    """

    by_seed: Dict[int, Dict[str, Dict[str, object]]] = {}
    for r in rows:
        try:
            seed = int(r["seed"])
        except Exception:
            continue
        by_seed.setdefault(seed, {})[str(r.get("mode"))] = r

    diffs: List[float] = []
    for seed in sorted(by_seed.keys()):
        row_a = by_seed[seed].get(a)
        row_b = by_seed[seed].get(b)
        if row_a is None or row_b is None:
            continue
        diffs.append(float(row_b.get(metric, 0.0)) - float(row_a.get(metric, 0.0)))

    n = int(len(diffs))
    if n == 0:
        return {"n": 0.0}

    diffs_arr = np.array(diffs, dtype=np.float64)
    mean = float(np.mean(diffs_arr))
    std = float(np.std(diffs_arr, ddof=1)) if n > 1 else 0.0
    sem = float(std / np.sqrt(n)) if n > 0 else 0.0
    t_score = float(mean / sem) if sem > 0 else float("-inf" if mean < 0 else ("inf" if mean > 0 else 0.0))

    p_left = float(NormalDist().cdf(t_score))
    try:
        from scipy import stats  # type: ignore

        if n > 1 and std > 0.0:
            p_left = float(stats.t.cdf(t_score, df=n - 1))
    except Exception:
        pass
    return {"n": float(n), "mean_diff": mean, "std_diff": std, "t_score": t_score, "p_left_approx": p_left}


def write_paired_diffs(rows: List[Dict[str, object]], *, metric: str, out_csv: str, a: str = "fixed", b: str = "dqn") -> None:
    by_seed: Dict[int, Dict[str, Dict[str, object]]] = {}
    for r in rows:
        try:
            seed = int(r["seed"])
        except Exception:
            continue
        by_seed.setdefault(seed, {})[str(r.get("mode"))] = r

    out_rows: List[Dict[str, object]] = []
    for seed in sorted(by_seed.keys()):
        row_a = by_seed[seed].get(a)
        row_b = by_seed[seed].get(b)
        if row_a is None or row_b is None:
            continue
        out_rows.append(
            {
                "seed": int(seed),
                "metric": metric,
                "mode_a": a,
                "mode_b": b,
                "diff": float(row_b.get(metric, 0.0)) - float(row_a.get(metric, 0.0)),
            }
        )
    write_csv(out_csv, out_rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate baselines vs DQN (2-green-phase control).")
    p.add_argument("--sumocfg", required=True)
    p.add_argument("--tls-id", required=True)
    p.add_argument("--routes-dir", required=True, help="Directory containing routes_seed{seed}.rou.xml")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--model", default="", help="Path to trained .keras model (optional)")
    p.add_argument("--gui", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=5400)
    p.add_argument("--green", type=int, default=33)
    p.add_argument("--yellow", type=int, default=6)
    p.add_argument("--outdir", default="", help="Output directory (default: runs/paper-eval-<timestamp>)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = args.outdir or os.path.join(PBL3_ROOT, "runs", f"paper-eval-{ts}")
    ensure_dir(outdir)
    plots_dir = ensure_dir(os.path.join(outdir, "plots"))

    dqn_model: Optional[tf.keras.Model] = None
    if args.model:
        dqn_model = tf.keras.models.load_model(args.model)

    rows: List[Dict[str, object]] = []
    modes = ["fixed", "heuristic"] + (["dqn"] if dqn_model is not None else [])

    for seed in args.seeds:
        route_path = os.path.join(args.routes_dir, f"routes_seed{int(seed)}.rou.xml")
        if not os.path.isfile(route_path):
            raise FileNotFoundError(f"Missing route file: {route_path}")

        for mode in modes:
            env = SumoTLEnv(
                sumocfg=args.sumocfg,
                tls_id=args.tls_id,
                seed=int(seed),
                gui=bool(args.gui),
                max_steps=int(args.max_steps),
                green_duration=int(args.green),
                yellow_duration=int(args.yellow),
                route_files=[route_path],
            )
            try:
                if mode == "fixed":
                    policy = FixedTimePolicy()
                elif mode == "heuristic":
                    policy = MaxQueuePolicy(env)
                elif mode == "dqn":
                    assert dqn_model is not None
                    policy = DQNPolicy(dqn_model)
                else:
                    raise RuntimeError(mode)

                out = run_episode(env, policy)
                out["seed"] = int(seed)
                out["mode"] = mode
                out["route_file"] = route_path
                rows.append(out)
                print(
                    f"seed={seed:03d} mode={mode:9s} "
                    f"NWT(abs)={out.get('nwt_abs',0):.1f} VQS={out.get('sum_intersection_queue',0):.1f} thr={out.get('throughput_junction',0):.1f}"
                )
            finally:
                env.close()

    runs_csv = os.path.join(outdir, "eval_runs.csv")
    summary_csv = os.path.join(outdir, "summary.csv")
    write_csv(runs_csv, rows)
    summarize(rows, out_csv=summary_csv)
    plot_eval(rows, os.path.join(plots_dir, "paper_eval.png"))

    if dqn_model is not None and any(str(r.get("mode")) == "fixed" for r in rows) and any(str(r.get("mode")) == "dqn" for r in rows):
        plot_paired_diffs(rows, os.path.join(plots_dir, "paper_paired_diffs.png"))
        analysis_txt = os.path.join(outdir, "paired_analysis.txt")
        diffs_nwt_csv = os.path.join(outdir, "paired_diffs_nwt_abs.csv")
        diffs_vqs_csv = os.path.join(outdir, "paired_diffs_vqs.csv")
        write_paired_diffs(rows, metric="nwt_abs", out_csv=diffs_nwt_csv)
        write_paired_diffs(rows, metric="sum_intersection_queue", out_csv=diffs_vqs_csv)

        a_nwt = paired_left_ttest(rows, metric="nwt_abs")
        a_vqs = paired_left_ttest(rows, metric="sum_intersection_queue")
        with open(analysis_txt, "w", encoding="utf-8") as handle:
            handle.write("Paired analysis (dqn - fixed), left-tailed: mean_diff < 0\n")
            handle.write(f"nwt_abs: {a_nwt}\n")
            handle.write(f"sum_intersection_queue: {a_vqs}\n")
            handle.write("p_left_approx uses Normal approximation (no SciPy).\n")

    print(f"Saved: {runs_csv}")
    print(f"Saved: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
