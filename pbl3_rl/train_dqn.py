import argparse
import csv
import os
import random
import sys
import time
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from env_sumo_tl import SumoTLEnv  # noqa: E402
from utils import generate_random_trips, read_sumocfg  # noqa: E402


def _parse_yellow_after(items: Sequence[str]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for raw in items:
        if ":" not in raw:
            raise ValueError(f"Invalid --yellow-after item {raw!r}. Expected 'green:yellow'")
        left, right = raw.split(":", 1)
        mapping[int(left)] = int(right)
    return mapping


def build_q_network(input_dim: int, num_actions: int, lr: float) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), dtype=tf.float32)
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_actions, activation=None)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.Huber(),
    )
    return model


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


def plot_training(log_rows: List[Dict[str, object]], out_png: str) -> None:
    import matplotlib.pyplot as plt

    if not log_rows:
        return
    episodes = [int(r["episode"]) for r in log_rows]
    rewards = [float(r["episode_reward"]) for r in log_rows]
    avg_queue = [float(r["avg_queue_time"]) for r in log_rows]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(episodes, rewards, label="episode_reward", color="tab:blue")
    ax1.set_xlabel("episode")
    ax1.set_ylabel("reward", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(episodes, avg_queue, label="avg_queue_time", color="tab:orange")
    ax2.set_ylabel("avg_queue_time", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_title("Training curve")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train DQN for SUMO traffic light control (green-phase actions only).")
    parser.add_argument("--sumocfg", required=True, help="Path to .sumocfg")
    parser.add_argument("--tls-id", required=True, help="Traffic light id")
    parser.add_argument("--outdir", default=os.path.join(THIS_DIR, "runs"), help="Output directory")
    parser.add_argument("--gui", type=int, default=0, help="1 to use sumo-gui")

    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--min-green", type=int, default=15)
    parser.add_argument("--max-green", type=int, default=60)
    parser.add_argument("--yellow-time", type=int, default=6)

    parser.add_argument("--green-phases", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--yellow-after", nargs="+", default=["0:1", "2:3", "4:5"])

    parser.add_argument("--demand", choices=["sumocfg", "randomtrips"], default="randomtrips")
    parser.add_argument("--train-seed-start", type=int, default=100)
    parser.add_argument("--insertion-density", type=float, default=12.0)
    parser.add_argument("--fringe-factor", type=float, default=5.0)
    parser.add_argument("--min-distance", type=float, default=300.0)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--start-training-after", type=int, default=500)
    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=10)

    args = parser.parse_args()

    yellow_after = _parse_yellow_after(args.yellow_after)

    run_name = time.strftime("dqn-%Y%m%d-%H%M%S")
    run_dir = ensure_dir(os.path.join(args.outdir, run_name))
    demand_dir = ensure_dir(os.path.join(run_dir, "demand"))
    models_dir = ensure_dir(os.path.join(run_dir, "models"))
    plots_dir = ensure_dir(os.path.join(run_dir, "plots"))

    cfg = read_sumocfg(args.sumocfg)
    net_file = str(cfg["net"])

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

    q_model: Optional[tf.keras.Model] = None
    target_model: Optional[tf.keras.Model] = None
    replay = deque(maxlen=int(args.buffer_size))

    log_rows: List[Dict[str, object]] = []
    global_step = 0

    eps = float(args.eps_start)
    eps_decay = (args.eps_start - args.eps_end) / max(1, int(args.eps_decay_steps))

    try:
        for episode in range(int(args.episodes)):
            seed = int(args.train_seed_start) + episode

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

            obs = env.reset(seed=seed, route_files=route_files)
            if q_model is None:
                q_model = build_q_network(env.obs_dim, env.num_actions, lr=float(args.lr))
                target_model = build_q_network(env.obs_dim, env.num_actions, lr=float(args.lr))
                target_model.set_weights(q_model.get_weights())

            done = False
            episode_reward = 0.0
            decision_steps = 0

            while not done:
                decision_steps += 1
                global_step += 1

                if random.random() < eps:
                    action = random.randrange(env.num_actions)
                else:
                    q_vals = q_model(obs.reshape(1, -1), training=False).numpy()[0]
                    action = int(np.argmax(q_vals))

                next_obs, reward, done, _info = env.step(action)
                episode_reward += float(reward)

                replay.append((obs, action, float(reward), next_obs, float(done)))
                obs = next_obs

                if len(replay) >= int(args.batch_size) and global_step >= int(args.start_training_after):
                    batch = random.sample(replay, int(args.batch_size))
                    states = np.stack([b[0] for b in batch]).astype(np.float32)
                    actions = np.asarray([b[1] for b in batch], dtype=np.int32)
                    rewards = np.asarray([b[2] for b in batch], dtype=np.float32)
                    next_states = np.stack([b[3] for b in batch]).astype(np.float32)
                    dones = np.asarray([b[4] for b in batch], dtype=np.float32)

                    next_q = target_model(next_states, training=False).numpy()
                    max_next_q = np.max(next_q, axis=1)
                    targets = rewards + float(args.gamma) * (1.0 - dones) * max_next_q

                    q_values = q_model(states, training=False).numpy()
                    q_values[np.arange(len(batch)), actions] = targets
                    q_model.train_on_batch(states, q_values)

                if global_step % int(args.target_update) == 0:
                    target_model.set_weights(q_model.get_weights())

                eps = max(float(args.eps_end), eps - eps_decay)

            metrics = env.get_episode_metrics()
            row = {
                "episode": episode,
                "seed": seed,
                "decision_steps": decision_steps,
                "episode_reward": episode_reward,
                "epsilon": eps,
                **metrics,
            }
            log_rows.append(row)

            if (episode + 1) % int(args.save_every) == 0:
                q_model.save(os.path.join(models_dir, f"dqn_ep{episode:04d}.keras"))

        q_model.save(os.path.join(models_dir, "dqn_final.keras"))
    finally:
        env.close()

    write_csv(os.path.join(run_dir, "train_log.csv"), log_rows)
    plot_training(log_rows, os.path.join(plots_dir, "training_curve.png"))

    print("run_dir:", run_dir)
    print("model:", os.path.join(run_dir, "models", "dqn_final.keras"))
    print("log:", os.path.join(run_dir, "train_log.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
