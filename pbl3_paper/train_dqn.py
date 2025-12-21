import argparse
import csv
import os
import random
import sys
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PBL3_ROOT = os.path.normpath(os.path.join(THIS_DIR, ".."))
for p in (THIS_DIR, PBL3_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

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


def build_q_model(state_dim: int, num_actions: int, hidden_sizes: Sequence[int], lr: float) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(state_dim,), dtype=tf.float32)
    x = inp
    for size in hidden_sizes:
        x = tf.keras.layers.Dense(int(size), activation="relu")(x)
    out = tf.keras.layers.Dense(num_actions, activation="linear")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)), loss="mse")
    return model


def plot_avg_series(avg_vals: List[float], out_png: str, title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    if not avg_vals:
        return
    episodes = list(range(len(avg_vals)))
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(episodes, avg_vals, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("episode")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DQN with protocol from experiment_config.yaml.")
    p.add_argument("--config", default=os.path.join(PBL3_ROOT, "experiment_config.yaml"))
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
    dqn = config.get("dqn", {})
    paths_cfg = config.get("paths", {})

    sumocfg = paths["sumocfg"]
    tls_id = paths["tls_id"]
    sim_seconds = int(exp.get("sim_seconds", 5400))
    episodes = int(exp.get("episodes", 100))
    repeats = int(exp.get("repeats", 3))
    vehicles = int(exp.get("vehicles", 1000))
    train_seed_start = int(exp.get("train_seed_start", 100))

    action_phase_indices = actions.get("action_phase_indices", [0, 2, 4, 6])

    results_dir = resolve_path(args.config, str(paths_cfg.get("results_dir", "results")))
    routes_root = resolve_path(args.config, str(paths_cfg.get("routes_dir", "results/routes")))

    ensure_dir(results_dir)
    training_dir = ensure_dir(os.path.join(results_dir, "training"))
    routes_root = ensure_dir(routes_root)

    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    net_file = read_net_from_sumocfg(sumocfg)
    turn_map = build_turn_map_from_net(net_file=net_file, tls_id=tls_id)

    hidden_sizes = dqn.get("hidden_sizes", [400, 400])
    gamma = float(dqn.get("gamma", 0.75))
    lr = float(dqn.get("lr", 0.001))
    replay_size = int(dqn.get("replay_size", 50000))
    batch_size = int(dqn.get("batch_size", 100))
    target_update = int(dqn.get("target_update_every", 10))
    eps_start = float(dqn.get("epsilon_start", 1.0))
    eps_end = float(dqn.get("epsilon_end", 0.01))
    eps_decay_eps = int(dqn.get("epsilon_decay_episodes", episodes))

    # Phase semantics verification (required, live via TraCI)
    env_check = SumoTLEnvCells(
        sumocfg=sumocfg,
        tls_id=tls_id,
        seed=int(train_seed_start),
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
            raise RuntimeError("Phase semantics verification failed. Fix TLS program before training.")
    finally:
        env_check.close()

    all_run_nwt: List[List[float]] = []
    all_run_vqs: List[List[float]] = []

    for run_idx in range(1, repeats + 1):
        run_rows: List[Dict[str, object]] = []
        run_routes_dir = ensure_dir(os.path.join(routes_root, f"run{run_idx}"))
        model_path = os.path.join(training_dir, f"run{run_idx}_model.keras")

        env = SumoTLEnvCells(
            sumocfg=sumocfg,
            tls_id=tls_id,
            seed=int(train_seed_start),
            gui=bool(args.gui),
            sim_seconds=sim_seconds,
            num_cells=10,
            action_phase_indices=action_phase_indices,
            timing_mode=str(timing.get("mode_timing", "KEEP_TLS_NATIVE")),
            green_step=int(timing.get("green_step", 10)),
            yellow_time=int(timing.get("yellow_time", 4)),
        )

        q_model = build_q_model(env.state_dim, env.num_actions, hidden_sizes, lr=float(lr))
        target_model = build_q_model(env.state_dim, env.num_actions, hidden_sizes, lr=float(lr))
        target_model.set_weights(q_model.get_weights())

        replay: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=int(replay_size))

        run_nwt: List[float] = []
        run_vqs: List[float] = []

        try:
            for ep in range(episodes):
                seed = train_seed_start + (run_idx - 1) * episodes + ep
                route_path = os.path.join(run_routes_dir, f"routes_seed{seed}.rou.xml")
                generate_routes_file(
                    out_route_file=route_path,
                    seed=seed,
                    turns=turn_map,
                    n_vehicles=int(vehicles),
                    end=int(sim_seconds),
                    weibull_shape=float(traffic.get("weibull_shape", 2.0)),
                    straight_ratio=float(traffic.get("straight_ratio", 0.75)),
                    turn_ratio=float(traffic.get("turn_ratio", 0.25)),
                    uturn_ratio=float(traffic.get("uturn_ratio", 0.0)),
                    allow_uturn=bool(traffic.get("allow_uturn", False)),
                    depart_speed="10",
                    vehicle_prefix="veh",
                )

                obs = env.reset(seed=seed, route_files=[route_path])
                done = False
                episode_reward = 0.0
                sum_neg_reward = 0.0
                vqs = 0.0
                decisions = 0

                if eps_decay_eps <= 1:
                    epsilon = eps_end
                else:
                    decay = min(1.0, float(ep) / float(eps_decay_eps - 1))
                    epsilon = eps_start + decay * (eps_end - eps_start)

                while not done:
                    if random.random() < epsilon:
                        action = random.randrange(env.num_actions)
                    else:
                        q = q_model.predict(obs[None, :], verbose=0)[0]
                        action = int(np.argmax(q))

                    next_obs, reward, done, info = env.step(action)
                    episode_reward += float(reward)
                    if float(reward) < 0.0:
                        sum_neg_reward += float(reward)
                    vqs = float(info.get("vqs", vqs))
                    decisions += 1

                    replay.append((obs, int(action), float(reward), next_obs, bool(done)))
                    obs = next_obs

                    if len(replay) >= int(batch_size):
                        batch = random.sample(replay, int(batch_size))
                        states = np.stack([b[0] for b in batch]).astype(np.float32)
                        actions_b = np.array([b[1] for b in batch], dtype=np.int32)
                        rewards = np.array([b[2] for b in batch], dtype=np.float32)
                        next_states = np.stack([b[3] for b in batch]).astype(np.float32)
                        dones = np.array([b[4] for b in batch], dtype=np.float32)

                        q_next = target_model.predict(next_states, verbose=0)
                        max_next = np.max(q_next, axis=1)
                        targets = rewards + (1.0 - dones) * float(gamma) * max_next

                        q_pred = q_model.predict(states, verbose=0)
                        q_pred[np.arange(len(batch)), actions_b] = targets
                        q_model.fit(states, q_pred, batch_size=len(batch), verbose=0)

                if (ep + 1) % int(target_update) == 0:
                    target_model.set_weights(q_model.get_weights())

                run_rows.append(
                    {
                        "run": int(run_idx),
                        "episode": int(ep),
                        "seed": int(seed),
                        "epsilon": float(epsilon),
                        "episode_reward": float(episode_reward),
                        "nwt": float(sum_neg_reward),
                        "nwt_abs": float(abs(sum_neg_reward)),
                        "vqs": float(vqs),
                        "decisions": int(decisions),
                    }
                )
                run_nwt.append(float(sum_neg_reward))
                run_vqs.append(float(vqs))

                print(
                    f"run {run_idx} ep {ep:03d} seed={seed} eps={epsilon:.3f} "
                    f"nwt={sum_neg_reward:.1f} vqs={vqs:.1f}"
                )
        finally:
            env.close()

        q_model.save(model_path)
        run_csv = os.path.join(training_dir, f"run{run_idx}.csv")
        write_csv(run_csv, run_rows)
        print(f"Saved: {run_csv}")
        print(f"Saved: {model_path}")

        all_run_nwt.append(run_nwt)
        all_run_vqs.append(run_vqs)

    # Average across runs (per episode)
    avg_nwt = [float(np.mean([run[ep] for run in all_run_nwt])) for ep in range(episodes)]
    avg_vqs = [float(np.mean([run[ep] for run in all_run_vqs])) for ep in range(episodes)]

    plot_avg_series(avg_nwt, os.path.join(training_dir, "avg_nwt.png"), "Avg cumulative negative wait time", "nwt")
    plot_avg_series(avg_vqs, os.path.join(training_dir, "avg_vqs.png"), "Avg cumulative vehicle queue size", "vqs")
    print(f"Saved: {os.path.join(training_dir, 'avg_nwt.png')}")
    print(f"Saved: {os.path.join(training_dir, 'avg_vqs.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
