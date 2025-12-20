import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from env_sumo_tl import SumoTLEnv  # noqa: E402
from utils import (  # noqa: E402
    build_sumo_cmd,
    generate_random_trips,
    get_controlled_lanes,
    get_total_queue,
    get_waiting_time_stats,
    read_sumocfg,
    resolve_sumo_binary,
    safe_traci_close,
)

import traci  # noqa: E402


def _parse_yellow_after(items: Sequence[str]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for raw in items:
        if ":" not in raw:
            raise ValueError(f"Invalid --yellow-after item {raw!r}. Expected 'green:yellow'")
        left, right = raw.split(":", 1)
        mapping[int(left)] = int(right)
    return mapping


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


def run_fixed_time(
    *,
    sumocfg: str,
    tls_id: str,
    seed: int,
    route_files: Optional[Sequence[str]],
    max_steps: int,
    gui: bool,
    measure_every: int = 15,
) -> Dict[str, float]:
    sumo_binary = resolve_sumo_binary(gui=gui)
    cmd = build_sumo_cmd(
        sumo_binary=sumo_binary,
        sumocfg=sumocfg,
        seed=seed,
        route_files=route_files,
        max_steps=max_steps,
    )

    traci.start(cmd)
    try:
        lanes = get_controlled_lanes(tls_id)
        if not lanes:
            raise RuntimeError(f"tls_id={tls_id!r} has no controlled lanes")
        lanes_set = set(lanes)

        sim_time = 0
        steps = 0
        queue_time_sum = 0.0
        arrived_total = 0
        avg_wait_samples: List[float] = []
        max_wait_samples: List[float] = []

        while True:
            traci.simulationStep()
            sim_time += 1
            steps += 1
            arrived_total += int(traci.simulation.getArrivedNumber())
            queue_time_sum += get_total_queue(lanes)

            if steps % int(measure_every) == 0:
                avg_wait, max_wait = get_waiting_time_stats(lanes_set)
                avg_wait_samples.append(avg_wait)
                max_wait_samples.append(max_wait)

            if sim_time >= int(max_steps):
                break
            if traci.simulation.getMinExpectedNumber() <= 0:
                break

        avg_queue_time = float(queue_time_sum / max(1, steps))
        avg_wait = float(sum(avg_wait_samples) / len(avg_wait_samples)) if avg_wait_samples else 0.0
        max_wait = float(max(max_wait_samples)) if max_wait_samples else 0.0
        score = -avg_queue_time - 0.2 * max_wait

        return {
            "score": float(score),
            "sim_time": float(sim_time),
            "steps": float(steps),
            "throughput": float(arrived_total),
            "avg_queue_time": float(avg_queue_time),
            "avg_wait": float(avg_wait),
            "max_wait": float(max_wait),
        }
    finally:
        safe_traci_close()


def choose_max_queue_action(env: SumoTLEnv) -> int:
    best_action = 0
    best_queue = -1.0
    for action_idx, served_lanes in enumerate(env.action_served_lanes):
        q = float(sum(traci.lane.getLastStepHaltingNumber(lane) for lane in served_lanes))
        if q > best_queue:
            best_queue = q
            best_action = action_idx
    return best_action


def run_controlled_episode(
    *,
    mode: str,
    env: SumoTLEnv,
    seed: int,
    route_files: Optional[Sequence[str]],
    model: Optional[tf.keras.Model] = None,
) -> Dict[str, float]:
    obs = env.reset(seed=seed, route_files=route_files)
    done = False
    total_reward = 0.0

    while not done:
        if mode == "heuristic":
            action = choose_max_queue_action(env)
        else:
            q_vals = model(obs.reshape(1, -1), training=False).numpy()[0]
            action = int(np.argmax(q_vals))

        obs, reward, done, _info = env.step(action)
        total_reward += float(reward)

    metrics = env.get_episode_metrics()
    score = -float(metrics["avg_queue_time"]) - 0.2 * float(metrics["max_wait"])
    return {"score": float(score), "total_reward": float(total_reward), **metrics}


def summarize(rows: List[Dict[str, object]], modes: Sequence[str], keys: Sequence[str]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for mode in modes:
        subset = [r for r in rows if r["mode"] == mode]
        if not subset:
            continue
        row: Dict[str, object] = {"mode": mode, "n": len(subset)}
        for key in keys:
            vals = np.asarray([float(r[key]) for r in subset], dtype=np.float32)
            row[f"{key}_mean"] = float(np.mean(vals))
            row[f"{key}_std"] = float(np.std(vals))
        out.append(row)
    return out


def plot_comparison(eval_rows: List[Dict[str, object]], summary_rows: List[Dict[str, object]], outdir: str) -> None:
    import matplotlib.pyplot as plt

    modes = [r["mode"] for r in summary_rows]
    avg_queue = [float(r["avg_queue_time_mean"]) for r in summary_rows]
    avg_wait = [float(r["avg_wait_mean"]) for r in summary_rows]
    throughput = [float(r["throughput_mean"]) for r in summary_rows]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].bar(modes, avg_queue, color="tab:orange")
    axes[0].set_title("avg_queue_time (mean)")
    axes[1].bar(modes, avg_wait, color="tab:green")
    axes[1].set_title("avg_wait (mean)")
    axes[2].bar(modes, throughput, color="tab:blue")
    axes[2].set_title("throughput (mean)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "plots", "comparison_bar.png"), dpi=150)
    plt.close(fig)

    seeds = sorted(set(int(r["seed"]) for r in eval_rows))
    fig, ax = plt.subplots(figsize=(10, 4))
    for mode in modes:
        ys = []
        for seed in seeds:
            match = next(r for r in eval_rows if r["mode"] == mode and int(r["seed"]) == seed)
            ys.append(float(match["avg_queue_time"]))
        ax.plot(seeds, ys, marker="o", label=mode)
    ax.set_xlabel("seed")
    ax.set_ylabel("avg_queue_time")
    ax.set_title("Per-seed avg_queue_time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "plots", "avg_queue_time_by_seed.png"), dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate fixed-time vs heuristic vs DQN on the same seeds.")
    parser.add_argument("--sumocfg", required=True, help="Path to .sumocfg")
    parser.add_argument("--tls-id", required=True, help="Traffic light id")
    parser.add_argument("--model", default=None, help="Path to trained Keras model (.keras). If omitted, skips DQN.")
    parser.add_argument("--outdir", default=os.path.join(THIS_DIR, "runs"), help="Output directory")
    parser.add_argument("--gui", type=int, default=0)

    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--min-green", type=int, default=15)
    parser.add_argument("--max-green", type=int, default=60)
    parser.add_argument("--yellow-time", type=int, default=6)

    parser.add_argument("--green-phases", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--yellow-after", nargs="+", default=["0:1", "2:3", "4:5"])

    parser.add_argument("--demand", choices=["sumocfg", "randomtrips"], default="randomtrips")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--insertion-density", type=float, default=12.0)
    parser.add_argument("--fringe-factor", type=float, default=5.0)
    parser.add_argument("--min-distance", type=float, default=300.0)

    args = parser.parse_args()

    yellow_after = _parse_yellow_after(args.yellow_after)

    run_name = time.strftime("eval-%Y%m%d-%H%M%S")
    run_dir = ensure_dir(os.path.join(args.outdir, run_name))
    demand_dir = ensure_dir(os.path.join(run_dir, "demand"))
    plots_dir = ensure_dir(os.path.join(run_dir, "plots"))

    cfg = read_sumocfg(args.sumocfg)
    net_file = str(cfg["net"])

    model = tf.keras.models.load_model(args.model) if args.model else None

    eval_rows: List[Dict[str, object]] = []

    for seed in args.seeds:
        seed = int(seed)
        route_files = None
        if args.demand == "randomtrips":
            trip_file = os.path.join(demand_dir, f"trips_seed{seed}.xml")
            if not os.path.isfile(trip_file):
                generate_random_trips(
                    net_file=net_file,
                    out_trip_file=trip_file,
                    seed=seed,
                    end_time=args.max_steps,
                    insertion_density=args.insertion_density,
                    fringe_factor=args.fringe_factor,
                    min_distance=args.min_distance,
                )
            route_files = [trip_file]

        fixed_metrics = run_fixed_time(
            sumocfg=args.sumocfg,
            tls_id=args.tls_id,
            seed=seed,
            route_files=route_files,
            max_steps=args.max_steps,
            gui=bool(args.gui),
            measure_every=args.min_green,
        )
        eval_rows.append({"seed": seed, "mode": "fixed", **fixed_metrics})

        env = SumoTLEnv(
            sumocfg=args.sumocfg,
            tls_id=args.tls_id,
            green_phases=args.green_phases,
            yellow_after=yellow_after,
            yellow_time=args.yellow_time,
            min_green=args.min_green,
            max_green=args.max_green,
            max_steps=args.max_steps,
            gui=bool(args.gui),
        )
        try:
            heur_metrics = run_controlled_episode(mode="heuristic", env=env, seed=seed, route_files=route_files)
            eval_rows.append({"seed": seed, "mode": "heuristic", **heur_metrics})
            if model is not None:
                dqn_metrics = run_controlled_episode(mode="dqn", env=env, seed=seed, route_files=route_files, model=model)
                eval_rows.append({"seed": seed, "mode": "dqn", **dqn_metrics})
        finally:
            env.close()

    write_csv(os.path.join(run_dir, "eval_results.csv"), eval_rows)

    modes = ["fixed", "heuristic"] + (["dqn"] if model is not None else [])
    keys = ["score", "avg_queue_time", "avg_wait", "max_wait", "throughput"]
    summary_rows = summarize(eval_rows, modes=modes, keys=keys)
    write_csv(os.path.join(run_dir, "summary.csv"), summary_rows)
    plot_comparison(eval_rows, summary_rows, run_dir)

    print("run_dir:", run_dir)
    print("results:", os.path.join(run_dir, "eval_results.csv"))
    print("summary:", os.path.join(run_dir, "summary.csv"))
    print("plots:", plots_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
