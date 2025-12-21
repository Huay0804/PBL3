import argparse
import csv
import os
import random
import sys
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PBL3_ROOT = os.path.normpath(os.path.join(THIS_DIR, ".."))
for p in (THIS_DIR, PBL3_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from env_sumo_tl import SumoTLEnv  # noqa: E402
from tools.gen_routes_weibull_paper import (  # noqa: E402
    build_turn_map_from_net,
    generate_routes_file,
    read_net_from_sumocfg,
)


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


def build_q_model(state_dim: int, num_actions: int, lr: float) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(state_dim,), dtype=tf.float32)
    x = tf.keras.layers.Dense(400, activation="relu")(inp)
    x = tf.keras.layers.Dense(400, activation="relu")(x)
    out = tf.keras.layers.Dense(num_actions, activation="linear")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)), loss="mse")
    return model


def plot_training(log_rows: List[Dict[str, object]], out_png: str) -> None:
    import matplotlib.pyplot as plt

    if not log_rows:
        return
    episodes = [int(r["episode"]) for r in log_rows]
    # Paper-style metrics:
    # - "cumulative negative wait time" ~= sum of negative rewards per episode
    # - "cumulative intersection queue size" ~= sum of queue at each decision step
    nwt = [float(r.get("sum_neg_reward", 0.0)) for r in log_rows]
    vqs = [float(r.get("sum_intersection_queue", 0.0)) for r in log_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(episodes, nwt, color="tab:blue", linewidth=1.5)
    ax1.set_title("Cumulative negative wait times across episodes")
    ax1.set_xlabel("episode")
    ax1.set_ylabel("cumulative negative wait time")
    ax1.grid(True, alpha=0.3)

    ax2.plot(episodes, vqs, color="tab:purple", linewidth=1.5)
    ax2.set_title("Cumulative intersection queue size across episodes")
    ax2.set_xlabel("episode")
    ax2.set_ylabel("cumulative intersection queue size")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DQN for 1 TLS junction (2 green actions, 40-cell binary state, delta-wait reward)."
    )
    p.add_argument("--sumocfg", required=True, help="Path to .sumocfg")
    p.add_argument("--tls-id", required=True, help="TLS id (e.g., GS_420249146)")
    p.add_argument("--gui", type=int, default=0, help="1 to show sumo-gui")

    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=5400, help="Episode length in seconds (default: 5400)")
    p.add_argument("--green", type=int, default=33, help="Green duration per decision (default: 33)")
    p.add_argument("--yellow", type=int, default=6, help="Yellow duration on switch (default: 6)")

    p.add_argument("--vehicles", type=int, default=1000, help="Vehicles per episode route file (default: 1000)")
    p.add_argument("--weibull-shape", type=float, default=2.0)
    p.add_argument("--straight-prob", type=float, default=0.75)
    p.add_argument("--train-seed-start", type=int, default=100, help="First seed for route generation (default: 100)")

    p.add_argument("--gamma", type=float, default=0.75)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--replay", type=int, default=50000)
    p.add_argument("--target-update", type=int, default=10, help="Update target net every N episodes (default: 10)")

    p.add_argument("--outdir", default="", help="Output directory (default: runs/paper-dqn-<timestamp>)")
    p.add_argument("--save-every", type=int, default=10, help="Save model every N episodes (default: 10)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = args.outdir or os.path.join(PBL3_ROOT, "runs", f"paper-dqn-{ts}")
    models_dir = ensure_dir(os.path.join(outdir, "models"))
    plots_dir = ensure_dir(os.path.join(outdir, "plots"))
    routes_dir = ensure_dir(os.path.join(outdir, "routes"))

    net_file = read_net_from_sumocfg(args.sumocfg)
    turn_map = build_turn_map_from_net(net_file=net_file, tls_id=args.tls_id)

    env = SumoTLEnv(
        sumocfg=args.sumocfg,
        tls_id=args.tls_id,
        gui=bool(args.gui),
        max_steps=int(args.max_steps),
        green_duration=int(args.green),
        yellow_duration=int(args.yellow),
    )

    q_model = build_q_model(env.state_dim, env.num_actions, lr=float(args.lr))
    target_model = build_q_model(env.state_dim, env.num_actions, lr=float(args.lr))
    target_model.set_weights(q_model.get_weights())

    replay: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=int(args.replay))
    log_rows: List[Dict[str, object]] = []

    print(f"outdir: {outdir}")
    print(f"state_dim: {env.state_dim} actions: {env.num_actions} ({', '.join(env.ACTION_NAMES)})")

    try:
        for ep in range(int(args.episodes)):
            seed = int(args.train_seed_start) + int(ep)
            route_path = os.path.join(routes_dir, f"routes_seed{seed}.rou.xml")
            generate_routes_file(
                out_route_file=route_path,
                seed=seed,
                turns=turn_map,
                n_vehicles=int(args.vehicles),
                end=int(args.max_steps),
                weibull_shape=float(args.weibull_shape),
                straight_prob=float(args.straight_prob),
                depart_speed="10",
                vehicle_prefix="veh",
            )

            obs = env.reset(seed=seed, route_files=[route_path])
            done = False
            episode_reward = 0.0
            sum_neg_reward = 0.0
            sum_intersection_queue = 0.0
            decisions = 0

            epsilon = max(0.0, 1.0 - (float(ep) / float(max(1, int(args.episodes)))))

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
                sum_intersection_queue += float(info.get("sum_queue", 0.0))
                decisions += 1

                replay.append((obs, int(action), float(reward), next_obs, bool(done)))
                obs = next_obs

                if len(replay) >= int(args.batch):
                    batch = random.sample(replay, int(args.batch))
                    states = np.stack([b[0] for b in batch]).astype(np.float32)
                    actions = np.array([b[1] for b in batch], dtype=np.int32)
                    rewards = np.array([b[2] for b in batch], dtype=np.float32)
                    next_states = np.stack([b[3] for b in batch]).astype(np.float32)
                    dones = np.array([b[4] for b in batch], dtype=np.float32)

                    q_next = target_model.predict(next_states, verbose=0)
                    max_next = np.max(q_next, axis=1)
                    targets = rewards + (1.0 - dones) * float(args.gamma) * max_next

                    q_pred = q_model.predict(states, verbose=0)
                    q_pred[np.arange(len(batch)), actions] = targets
                    q_model.fit(states, q_pred, batch_size=len(batch), verbose=0)

            # Target network sync (paper-style, every N episodes)
            if (ep + 1) % int(args.target_update) == 0:
                target_model.set_weights(q_model.get_weights())

            row = dict(info)
            row.update(
                {
                    "episode": int(ep),
                    "seed": int(seed),
                    "epsilon": float(epsilon),
                    "episode_reward": float(episode_reward),
                    "sum_neg_reward": float(sum_neg_reward),
                    "nwt_abs": float(abs(sum_neg_reward)),
                    "sum_intersection_queue": float(sum_intersection_queue),
                    "decisions": int(decisions),
                }
            )
            log_rows.append(row)

            if (ep + 1) % int(args.save_every) == 0 or (ep + 1) == int(args.episodes):
                q_model.save(os.path.join(models_dir, f"dqn_ep{ep}.keras"))
                q_model.save(os.path.join(models_dir, "dqn_latest.keras"))

            print(
                f"ep {ep:03d} seed={seed} eps={epsilon:.3f} "
                f"NWT={sum_neg_reward:.1f} VQS={sum_intersection_queue:.1f} "
                f"avgQ={row.get('avg_queue',0):.3f} avgW={row.get('avg_wait',0):.3f} thr={row.get('throughput_junction',0):.1f}"
            )

    finally:
        env.close()

    q_model.save(os.path.join(models_dir, "dqn_final.keras"))
    write_csv(os.path.join(outdir, "train_log.csv"), log_rows)
    plot_training(log_rows, os.path.join(plots_dir, "training_paper.png"))
    print(f"Saved: {os.path.join(models_dir, 'dqn_final.keras')}")
    print(f"Saved: {os.path.join(outdir, 'train_log.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
