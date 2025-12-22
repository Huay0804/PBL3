import argparse
import csv
import os
import sys
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PBL3_ROOT = os.path.normpath(os.path.join(THIS_DIR, ".."))
for p in (THIS_DIR, PBL3_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from baseline_fds import run_fds_episode  # noqa: E402
from env_sumo_cells import SumoTLEnvCells  # noqa: E402
from sumo_lane_cells import (  # noqa: E402
    load_experiment_config,
    read_sumocfg,
    resolve_config_paths,
    resolve_path,
)
from tools.gen_routes import build_turn_map_from_net, generate_routes_file, read_net_from_sumocfg  # noqa: E402


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


def _gaussian_kde(xs: np.ndarray, samples: np.ndarray) -> np.ndarray:
    if samples.size == 0:
        return np.zeros_like(xs)
    if samples.size == 1:
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


def _plot_metric(
    ax,
    *,
    vals_fds: List[float],
    vals_adap: List[float],
    title: str,
) -> None:
    if not vals_fds or not vals_adap:
        return
    fds = np.array(vals_fds, dtype=np.float64)
    adap = np.array(vals_adap, dtype=np.float64)
    xmin = float(min(np.min(fds), np.min(adap)))
    xmax = float(max(np.max(fds), np.max(adap)))
    xs = np.linspace(xmin, xmax, 300)

    color_adap = "tab:blue"
    color_fds = "tab:orange"

    ax.hist(adap, bins=15, density=True, alpha=0.35, color=color_adap, label="Adaptive TLCS")
    ax.hist(fds, bins=15, density=True, alpha=0.35, color=color_fds, label="FDS TLCS")
    ax.plot(xs, _gaussian_kde(xs, adap), color=color_adap, linewidth=1.6)
    ax.plot(xs, _gaussian_kde(xs, fds), color=color_fds, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel(title)
    ax.set_ylabel("density")
    ax.grid(False)
    ax.legend()


def plot_hist_compare(vals_fds: List[float], vals_adap: List[float], title: str, out_png: str) -> None:
    import matplotlib.pyplot as plt

    if not vals_fds or not vals_adap:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    _plot_metric(ax, vals_fds=vals_fds, vals_adap=vals_adap, title=title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_eval_panels(
    *,
    fds_nwt_abs: List[float],
    adap_nwt_abs: List[float],
    fds_vqs: List[float],
    adap_vqs: List[float],
    out_png: str,
) -> None:
    import matplotlib.pyplot as plt

    if not fds_nwt_abs or not adap_nwt_abs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _plot_metric(
        axes[0],
        vals_fds=fds_nwt_abs,
        vals_adap=adap_nwt_abs,
        title="Cumulative Negative Wait Time",
    )
    _plot_metric(
        axes[1],
        vals_fds=fds_vqs,
        vals_adap=adap_vqs,
        title="Cumulative Vehicle Queue Size",
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def paired_left_ttest(diffs: List[float]) -> Dict[str, float]:
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
    return {"n": float(n), "mean_diff": mean, "std_diff": std, "t_score": t_score, "p_value": p_left}


def run_dqn_episode(env: SumoTLEnvCells, model: tf.keras.Model) -> Dict[str, float]:
    obs = env.reset()
    done = False
    sum_neg_reward = 0.0
    vqs = 0.0
    w_t = 0.0
    while not done:
        q = model.predict(obs[None, :], verbose=0)[0]
        action = int(np.argmax(q))
        obs, reward, done, info = env.step(action)
        if reward < 0:
            sum_neg_reward += float(reward)
        vqs = float(info.get("vqs", vqs))
        w_t = float(info.get("w_t", w_t))
    return {"nwt": float(sum_neg_reward), "nwt_abs": float(abs(sum_neg_reward)), "vqs": float(vqs), "w_t": float(w_t)}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate FDS vs Adaptive policy using experiment_config.yaml.")
    p.add_argument("--config", default=os.path.join(PBL3_ROOT, "experiment_config.yaml"))
    p.add_argument("--model", default="", help="Path to trained .keras model")
    p.add_argument("--gui", type=int, default=0)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_experiment_config(args.config)
    paths = resolve_config_paths(args.config, config)

    exp = config.get("experiment", {})
    traffic = config.get("traffic", {})
    actions = config.get("actions", {})
    timing = config.get("timing", {})
    paths_cfg = config.get("paths", {})

    sumocfg = paths["sumocfg"]
    tls_id = paths["tls_id"]
    sim_seconds = int(exp.get("sim_seconds", 5400))
    eval_sims = int(exp.get("eval_sims", 100))

    action_phase_indices = actions.get("action_phase_indices", [0, 2, 4, 6])
    depart_lane = str(traffic.get("depart_lane", "best"))
    depart_speed = str(traffic.get("eval_depart_speed", "auto"))

    results_dir = resolve_path(args.config, str(paths_cfg.get("results_dir", "results")))
    routes_root = resolve_path(args.config, str(paths_cfg.get("routes_dir", "results/routes")))
    eval_dir = ensure_dir(os.path.join(results_dir, "eval"))
    routes_eval_dir = ensure_dir(os.path.join(routes_root, "eval"))

    model_path = args.model.strip()
    if not model_path:
        model_path = os.path.join(results_dir, "training", f"run{int(exp.get('repeats', 3))}_model.keras")
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        raise RuntimeError(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)

    net_file = read_net_from_sumocfg(sumocfg)
    turn_map = build_turn_map_from_net(net_file=net_file, tls_id=tls_id)

    # Phase semantics verification (required, live via TraCI)
    env_check = SumoTLEnvCells(
        sumocfg=sumocfg,
        tls_id=tls_id,
        seed=0,
        gui=bool(args.gui),
        sim_seconds=sim_seconds,
        num_cells=10,
        action_phase_indices=action_phase_indices,
        timing_mode=str(timing.get("mode_timing", "KEEP_TLS_NATIVE")),
        green_step=int(timing.get("green_step", 10)),
        yellow_time=int(timing.get("yellow_time", 4)),
    )
    try:
        env_check.reset()
        report = env_check.verify_phase_semantics_live()
        print("phase_semantics_ok:", report.ok)
        for issue in report.issues:
            print("ISSUE:", issue)
        for warn in report.warnings:
            print("WARN:", warn)
        if not report.ok:
            raise RuntimeError("Phase semantics verification failed. Fix TLS program before evaluation.")
    finally:
        env_check.close()

    seeds = list(range(eval_sims))

    rows: List[Dict[str, object]] = []
    fds_nwt_abs: List[float] = []
    adap_nwt_abs: List[float] = []
    fds_vqs: List[float] = []
    adap_vqs: List[float] = []

    for seed in seeds:
        route_path = os.path.join(routes_eval_dir, f"routes_seed{int(seed)}.rou.xml")
        if not os.path.isfile(route_path):
            generate_routes_file(
                out_route_file=route_path,
                seed=int(seed),
                turns=turn_map,
                n_vehicles=int(exp.get("vehicles", 1000)),
                end=int(sim_seconds),
                weibull_shape=float(traffic.get("weibull_shape", 2.0)),
                straight_ratio=float(traffic.get("straight_ratio", 0.75)),
                turn_ratio=float(traffic.get("turn_ratio", 0.25)),
                uturn_ratio=float(traffic.get("uturn_ratio", 0.0)),
                allow_uturn=bool(traffic.get("allow_uturn", False)),
                depart_lane=str(depart_lane),
                depart_speed=str(depart_speed),
                vehicle_prefix="veh",
            )

        env_fds = SumoTLEnvCells(
            sumocfg=sumocfg,
            tls_id=tls_id,
            seed=int(seed),
            gui=bool(args.gui),
            sim_seconds=sim_seconds,
            num_cells=10,
            action_phase_indices=action_phase_indices,
            timing_mode=str(timing.get("mode_timing", "KEEP_TLS_NATIVE")),
            green_step=int(timing.get("green_step", 10)),
            yellow_time=int(timing.get("yellow_time", 4)),
            route_files=[route_path],
        )
        env_dqn = SumoTLEnvCells(
            sumocfg=sumocfg,
            tls_id=tls_id,
            seed=int(seed),
            gui=bool(args.gui),
            sim_seconds=sim_seconds,
            num_cells=10,
            action_phase_indices=action_phase_indices,
            timing_mode=str(timing.get("mode_timing", "KEEP_TLS_NATIVE")),
            green_step=int(timing.get("green_step", 10)),
            yellow_time=int(timing.get("yellow_time", 4)),
            route_files=[route_path],
        )

        try:
            fds_out = run_fds_episode(env_fds)
            dqn_out = run_dqn_episode(env_dqn, model)
        finally:
            env_fds.close()
            env_dqn.close()

        rows.append(
            {
                "seed": int(seed),
                "fds_nwt": float(fds_out["nwt"]),
                "fds_nwt_abs": float(fds_out["nwt_abs"]),
                "fds_vqs": float(fds_out["vqs"]),
                "adap_nwt": float(dqn_out["nwt"]),
                "adap_nwt_abs": float(dqn_out["nwt_abs"]),
                "adap_vqs": float(dqn_out["vqs"]),
            }
        )

        fds_nwt_abs.append(float(fds_out["nwt_abs"]))
        adap_nwt_abs.append(float(dqn_out["nwt_abs"]))
        fds_vqs.append(float(fds_out["vqs"]))
        adap_vqs.append(float(dqn_out["vqs"]))

        print(
            f"seed={seed:03d} fds_nwt_abs={fds_out['nwt_abs']:.1f} adap_nwt_abs={dqn_out['nwt_abs']:.1f} "
            f"fds_vqs={fds_out['vqs']:.1f} adap_vqs={dqn_out['vqs']:.1f}"
        )

    eval_csv = os.path.join(eval_dir, "eval.csv")
    write_csv(eval_csv, rows)

    plot_hist_compare(fds_nwt_abs, adap_nwt_abs, "Cumulative Negative Wait Time", os.path.join(eval_dir, "eval_nwt.png"))
    plot_hist_compare(fds_vqs, adap_vqs, "Cumulative Vehicle Queue Size", os.path.join(eval_dir, "eval_vqs.png"))
    plot_eval_panels(
        fds_nwt_abs=fds_nwt_abs,
        adap_nwt_abs=adap_nwt_abs,
        fds_vqs=fds_vqs,
        adap_vqs=adap_vqs,
        out_png=os.path.join(eval_dir, "eval_hist.png"),
    )

    diffs_nwt = [adap_nwt_abs[i] - fds_nwt_abs[i] for i in range(len(fds_nwt_abs))]
    diffs_vqs = [adap_vqs[i] - fds_vqs[i] for i in range(len(fds_vqs))]

    stats_nwt = paired_left_ttest(diffs_nwt)
    stats_vqs = paired_left_ttest(diffs_vqs)

    stats_txt = os.path.join(eval_dir, "stats.txt")
    with open(stats_txt, "w", encoding="utf-8") as handle:
        handle.write("Evaluation summary (means/std use nwt_abs and vqs)\n")
        handle.write(f"nwt_abs_fds_mean={np.mean(fds_nwt_abs):.3f} nwt_abs_fds_std={np.std(fds_nwt_abs):.3f}\n")
        handle.write(f"nwt_abs_adap_mean={np.mean(adap_nwt_abs):.3f} nwt_abs_adap_std={np.std(adap_nwt_abs):.3f}\n")
        handle.write(f"vqs_fds_mean={np.mean(fds_vqs):.3f} vqs_fds_std={np.std(fds_vqs):.3f}\n")
        handle.write(f"vqs_adap_mean={np.mean(adap_vqs):.3f} vqs_adap_std={np.std(adap_vqs):.3f}\n")
        handle.write("Paired left-tailed t-test (Adaptive - FDS): mean_diff < 0\n")
        handle.write(f"nwt_abs: {stats_nwt}\n")
        handle.write(f"vqs: {stats_vqs}\n")

    print(f"Saved: {eval_csv}")
    print(f"Saved: {stats_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
