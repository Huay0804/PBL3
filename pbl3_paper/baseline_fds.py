import argparse
import os
import sys
from typing import Dict, Optional, Sequence

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PBL3_ROOT = os.path.normpath(os.path.join(THIS_DIR, ".."))
for p in (THIS_DIR, PBL3_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from env_sumo_cells import SumoTLEnvCells  # noqa: E402
from sumo_lane_cells import (  # noqa: E402
    build_lane_groups,
    load_experiment_config,
    read_sumocfg,
    resolve_config_paths,
    verify_phase_semantics,
)


def run_fds_episode(env: SumoTLEnvCells) -> Dict[str, float]:
    obs = env.reset()
    done = False
    step_idx = 0
    action_cycle = list(range(env.num_actions))
    sum_neg_reward = 0.0
    vqs = 0.0
    w_t = 0.0

    while not done:
        action = action_cycle[step_idx % len(action_cycle)]
        obs, reward, done, info = env.step(action)
        if reward < 0:
            sum_neg_reward += float(reward)
        vqs = float(info.get("vqs", vqs))
        w_t = float(info.get("w_t", w_t))
        step_idx += 1

    return {
        "nwt": float(sum_neg_reward),
        "nwt_abs": float(abs(sum_neg_reward)),
        "vqs": float(vqs),
        "w_t": float(w_t),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run fixed-duration fixed-sequence (FDS) baseline.")
    p.add_argument("--config", default=os.path.join(PBL3_ROOT, "experiment_config.yaml"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gui", type=int, default=0)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_experiment_config(args.config)
    paths = resolve_config_paths(args.config, config)

    exp = config.get("experiment", {})
    timing = config.get("timing", {})
    actions = config.get("actions", {})

    sumocfg = paths["sumocfg"]
    tls_id = paths["tls_id"]

    net_file = str(read_sumocfg(sumocfg)["net"])
    lane_groups = build_lane_groups(net_file=net_file, tls_id=tls_id)
    report = verify_phase_semantics(lane_groups, actions.get("action_phase_indices", []))
    print("phase_semantics_ok:", report.ok)
    for issue in report.issues:
        print("ISSUE:", issue)
    for warn in report.warnings:
        print("WARN:", warn)

    env = SumoTLEnvCells(
        sumocfg=sumocfg,
        tls_id=tls_id,
        seed=int(args.seed),
        gui=bool(args.gui),
        sim_seconds=int(exp.get("sim_seconds", 5400)),
        num_cells=10,
        action_phase_indices=actions.get("action_phase_indices", [0, 2, 4, 6]),
        timing_mode=str(timing.get("mode_timing", "KEEP_TLS_NATIVE")),
        green_step=int(timing.get("green_step", 10)),
        yellow_time=int(timing.get("yellow_time", 4)),
    )
    try:
        out = run_fds_episode(env)
        print(f"seed={args.seed} nwt={out['nwt']:.1f} vqs={out['vqs']:.1f} w_t={out['w_t']:.1f}")
    finally:
        env.close()
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
