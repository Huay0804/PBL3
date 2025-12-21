import argparse
import os
import shutil
import sys
from contextlib import contextmanager
from typing import List, Optional, Sequence, Set

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PBL3_ROOT = os.path.normpath(os.path.join(THIS_DIR, ".."))
for p in (THIS_DIR, PBL3_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from pbl3_paper.sumo_lane_cells import (  # noqa: E402
    build_lane_groups,
    read_sumocfg,
    served_groups_for_phase,
    verify_phase_semantics,
)


def ensure_sumo_tools() -> str:
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        for candidate in ("sumo", "sumo.exe", "sumo-gui", "sumo-gui.exe"):
            resolved = shutil.which(candidate)
            if resolved:
                sumo_home = os.path.dirname(os.path.dirname(resolved))
                break

    if not sumo_home:
        raise RuntimeError(
            "SUMO_HOME not set and SUMO not found on PATH. Install SUMO and set SUMO_HOME or add SUMO/bin to PATH."
        )

    tools_path = os.path.join(sumo_home, "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)
    os.environ["SUMO_HOME"] = sumo_home
    return sumo_home


def resolve_sumo_binary(gui: bool) -> str:
    sumo_home = ensure_sumo_tools()
    names = ["sumo-gui.exe", "sumo-gui"] if gui else ["sumo.exe", "sumo"]
    for name in names:
        candidate = os.path.join(sumo_home, "bin", name)
        if os.path.isfile(candidate):
            return candidate
    for name in names:
        resolved = shutil.which(name)
        if resolved:
            return resolved
    raise RuntimeError("Could not find sumo binary. Check SUMO_HOME/bin or PATH.")


@contextmanager
def pushd(path: str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def infer_incoming_lanes(tls_id: str, traci_module) -> List[str]:
    controlled_links = traci_module.trafficlight.getControlledLinks(tls_id)
    lanes: Set[str] = set()
    for link in controlled_links:
        for conn in link:
            if not conn:
                continue
            from_lane = conn[0]
            if from_lane and not from_lane.startswith(":"):
                lanes.add(from_lane)
    return sorted(lanes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect TLS program and lane-groups (TR/LU) per arm.")
    parser.add_argument("--sumocfg", required=True, help="Path to .sumocfg")
    parser.add_argument("--tls-id", required=True, help="Traffic light id")
    parser.add_argument("--gui", type=int, default=0, help="1 to use sumo-gui")
    parser.add_argument("--expected-links", type=int, default=None, help="Expected controlled link count")
    args = parser.parse_args()

    sumocfg = os.path.abspath(args.sumocfg)
    if not os.path.isfile(sumocfg):
        raise RuntimeError(f"sumocfg not found: {sumocfg}")

    cfg = read_sumocfg(sumocfg)
    net_file = str(cfg["net"])
    lane_groups = build_lane_groups(net_file=net_file, tls_id=args.tls_id)

    sumo_bin = resolve_sumo_binary(gui=bool(args.gui))
    ensure_sumo_tools()
    import traci  # noqa: E402

    scenario_dir = os.path.dirname(sumocfg)
    with pushd(scenario_dir):
        cmd = [
            sumo_bin,
            "-c",
            sumocfg,
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
        ]
        traci.start(cmd)
        try:
            tls_ids = set(traci.trafficlight.getIDList())
            if args.tls_id not in tls_ids:
                raise RuntimeError(f"tls-id {args.tls_id!r} not found. Available: {', '.join(sorted(tls_ids))}")

            link_count = len(traci.trafficlight.getControlledLinks(args.tls_id))
            if args.expected_links is not None and int(args.expected_links) != link_count:
                print(f"WARNING: expected_links={args.expected_links} but controlled_links={link_count}")

            current_program = traci.trafficlight.getProgram(args.tls_id)
            current_phase = traci.trafficlight.getPhase(args.tls_id)
            remaining = traci.trafficlight.getNextSwitch(args.tls_id) - traci.simulation.getTime()

            print(f"sumocfg: {sumocfg}")
            print(f"tls_id:  {args.tls_id}")
            print(f"net:     {net_file}")
            print(f"sim_time: {traci.simulation.getTime()}")
            print(f"current_program: {current_program}")
            print(f"current_phase:   {current_phase}")
            print(f"remaining_phase_time: {remaining:.2f}s")
            print(f"controlled_links: {link_count}")

            incoming_lanes = infer_incoming_lanes(args.tls_id, traci)
            print(f"incoming_lanes({len(incoming_lanes)}):")
            for lane in incoming_lanes:
                print(f"  - {lane}")

            print("lane_groups (per arm):")
            for arm in lane_groups.arm_order:
                groups = lane_groups.groups.get(arm, {})
                tr = groups.get("TR", [])
                lu = groups.get("LU", [])
                print(f"  {arm}:")
                print(f"    TR: {', '.join(tr) if tr else '(none)'}")
                print(f"    LU: {', '.join(lu) if lu else '(none)'}")

            print("state_order (group -> 10 cells):")
            order = [f"{arm}_{grp}" for arm, grp in lane_groups.group_order]
            print(f"  {order}")
            print(f"state_dim = {len(order) * 10}")

            print("program_logics:")
            for idx, (dur, state) in enumerate(
                zip(lane_groups.program.phase_durations, lane_groups.program.phase_states)
            ):
                print(f"  phase {idx}: dur={dur:.1f} state_len={len(state)} state='{state}'")

            for ph in [0, 2, 4, 6]:
                if ph < len(lane_groups.program.phase_states):
                    served = served_groups_for_phase(
                        lane_groups.program.phase_states[ph], lane_groups.conns, lane_groups.lane_to_group
                    )
                    print(f"served_groups phase {ph}: {sorted(served)}")

            report = verify_phase_semantics(lane_groups, [0, 2, 4, 6])
            print(f"phase_semantics_ok: {report.ok}")
            for issue in report.issues:
                print(f"ISSUE: {issue}")
            for warn in report.warnings:
                print(f"WARN: {warn}")
        finally:
            try:
                traci.close(False)
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
