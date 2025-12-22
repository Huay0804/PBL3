import argparse
import os
import sys
import time
from typing import Optional, Sequence

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PBL3_ROOT = os.path.normpath(os.path.join(THIS_DIR, ".."))
for p in (THIS_DIR, PBL3_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from pbl3_paper.env_sumo_cells import SumoTLEnvCells  # noqa: E402
from pbl3_paper.sumo_lane_cells import (  # noqa: E402
    load_experiment_config,
    resolve_config_paths,
)
from tools.gen_routes import (  # noqa: E402
    build_turn_map_from_net,
    generate_routes_file,
    read_net_from_sumocfg,
)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a trained DQN model in SUMO-GUI with a fresh route file.")
    p.add_argument("--config", default=os.path.join(PBL3_ROOT, "experiment_config.yaml"))
    p.add_argument("--model", required=True, help="Path to trained .keras model")
    p.add_argument("--seed", type=int, default=None, help="Seed for route generation (default: time-based)")
    p.add_argument("--sim-seconds", type=int, default=None, help="Override sim duration")
    p.add_argument("--vehicles", type=int, default=None, help="Override vehicles per episode")
    p.add_argument("--log-every", type=int, default=60, help="Log interval in seconds (sim time)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config_path = os.path.abspath(args.config)
    cfg = load_experiment_config(config_path)
    paths = resolve_config_paths(config_path, cfg)

    sumocfg = paths["sumocfg"]
    tls_id = paths["tls_id"]

    exp = cfg.get("experiment", {})
    traffic = cfg.get("traffic", {})
    actions = cfg.get("actions", {})
    timing = cfg.get("timing", {})

    sim_seconds = int(args.sim_seconds or exp.get("sim_seconds", 5400))
    vehicles = int(args.vehicles or exp.get("vehicles", 1000))

    seed = int(args.seed) if args.seed is not None else int(time.time()) % 1000000

    net_file = read_net_from_sumocfg(sumocfg)
    turn_map = build_turn_map_from_net(net_file=net_file, tls_id=tls_id)

    routes_root = ensure_dir(os.path.join(PBL3_ROOT, "results", "routes", "vis"))
    route_path = os.path.join(routes_root, f"routes_vis_seed{seed}.rou.xml")
    generate_routes_file(
        out_route_file=route_path,
        seed=seed,
        turns=turn_map,
        n_vehicles=vehicles,
        end=sim_seconds,
        weibull_shape=float(traffic.get("weibull_shape", 2.0)),
        straight_ratio=float(traffic.get("straight_ratio", 0.75)),
        turn_ratio=float(traffic.get("turn_ratio", 0.25)),
        uturn_ratio=float(traffic.get("uturn_ratio", 0.0)),
        allow_uturn=bool(traffic.get("allow_uturn", False)),
        depart_lane=str(traffic.get("depart_lane", "best")),
        depart_speed=str(traffic.get("depart_speed", "auto")),
        vehicle_prefix="vis",
    )

    model_path = os.path.abspath(args.model)
    if not os.path.isfile(model_path):
        raise RuntimeError(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)

    action_phase_indices = actions.get("action_phase_indices", [0, 2, 4, 6])
    env = SumoTLEnvCells(
        sumocfg=sumocfg,
        tls_id=tls_id,
        seed=seed,
        gui=True,
        sim_seconds=sim_seconds,
        num_cells=10,
        action_phase_indices=action_phase_indices,
        timing_mode=str(timing.get("mode_timing", "KEEP_TLS_NATIVE")),
        green_step=int(timing.get("green_step", 10)),
        yellow_time=int(timing.get("yellow_time", 4)),
        route_files=[route_path],
    )

    obs = env.reset(seed=seed, route_files=[route_path])
    done = False
    next_log_time = 0
    last_info = {}
    try:
        while not done:
            q = model.predict(obs[None, :], verbose=0)[0]
            action = int(np.argmax(q))
            obs, reward, done, info = env.step(action)
            last_info = info
            sim_time = float(info.get("time", 0.0))
            if sim_time >= next_log_time:
                print(
                    f"time={sim_time:6.0f}s action={action} phase={int(info.get('green_phase', -1))} "
                    f"w_t={info.get('w_t', 0.0):.1f} vqs={info.get('vqs', 0.0):.1f} "
                    f"sum_queue={info.get('sum_queue', 0.0):.1f} reward={reward:.1f}",
                    flush=True,
                )
                next_log_time += int(args.log_every)
    finally:
        env.close()

    if last_info:
        print(
            f"done: w_t={last_info.get('w_t', 0.0):.1f} vqs={last_info.get('vqs', 0.0):.1f} "
            f"throughput={last_info.get('throughput_junction', 0.0):.0f}",
            flush=True,
        )
    print(f"route_file: {route_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
