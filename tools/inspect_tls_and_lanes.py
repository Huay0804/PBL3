import argparse
import os
import shutil
import sys
from contextlib import contextmanager
from typing import Iterable, List, Optional, Sequence, Set


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


def unique_sorted(values: Iterable[str]) -> List[str]:
    return sorted(set(values))


def infer_incoming_lanes_from_links(controlled_links: Sequence[Sequence[Sequence[str]]]) -> List[str]:
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
    parser = argparse.ArgumentParser(description="Inspect SUMO TLS program logics and incoming lanes via TraCI.")
    parser.add_argument("--sumocfg", required=True, help="Path to .sumocfg")
    parser.add_argument("--tls-id", required=True, help="Traffic light id")
    parser.add_argument("--gui", type=int, default=0, help="1 to use sumo-gui")
    parser.add_argument("--expected-links", type=int, default=17, help="Expected controlled links (default: 17)")
    args = parser.parse_args()

    sumocfg = os.path.abspath(args.sumocfg)
    if not os.path.isfile(sumocfg):
        raise RuntimeError(f"sumocfg not found: {sumocfg}")

    sumo_bin = resolve_sumo_binary(gui=bool(args.gui))

    ensure_sumo_tools()
    import traci  # noqa: E402

    scenario_dir = os.path.dirname(sumocfg)
    with pushd(scenario_dir):
        traci.start([sumo_bin, "-c", sumocfg, "--no-step-log", "true"])
        try:
            tls_ids = set(traci.trafficlight.getIDList())
            if args.tls_id not in tls_ids:
                raise RuntimeError(f"tls-id {args.tls_id!r} not found. Available: {', '.join(sorted(tls_ids))}")

            current_program = traci.trafficlight.getProgram(args.tls_id)
            current_phase = traci.trafficlight.getPhase(args.tls_id)
            sim_time = traci.simulation.getTime()
            next_switch = traci.trafficlight.getNextSwitch(args.tls_id)
            remaining = float(next_switch - sim_time)

            print(f"sumocfg: {sumocfg}")
            print(f"tls_id:  {args.tls_id}")
            print(f"sim_time: {sim_time}")
            print(f"current_program: {current_program}")
            print(f"current_phase:   {current_phase}")
            print(f"remaining_phase_time: {remaining:.2f}s")

            controlled_links = traci.trafficlight.getControlledLinks(args.tls_id)
            link_count = len(controlled_links)
            print(f"controlled_links: {link_count}")

            if args.expected_links is not None and int(args.expected_links) != link_count:
                print(f"WARNING: expected_links={args.expected_links} but controlled_links={link_count}")

            lanes = infer_incoming_lanes_from_links(controlled_links)
            if not lanes:
                lanes = unique_sorted(
                    [l for l in traci.trafficlight.getControlledLanes(args.tls_id) if l and not l.startswith(":")]
                )
            print(f"incoming_lanes({len(lanes)}):")
            for lane in lanes:
                print(f"  - {lane}")

            print("program_logics:")
            logics = traci.trafficlight.getAllProgramLogics(args.tls_id)
            if not logics:
                print("  (no program logics returned)")
                return 0

            for logic in logics:
                pid = getattr(logic, "programID", None) or getattr(logic, "programId", None) or "?"
                phases = getattr(logic, "phases", []) or []
                active = " (active)" if str(pid) == str(current_program) else ""
                print(f"- programID={pid}{active} phases={len(phases)}")
                for idx, phase in enumerate(phases):
                    dur = getattr(phase, "duration", None)
                    state = getattr(phase, "state", "") or ""
                    print(f"    phase {idx}: dur={dur} state_len={len(state)} state={state!r}")

                    if len(state) != link_count:
                        raise RuntimeError(
                            f"Sanity check failed: phase {idx} state_len={len(state)} != controlled_links={link_count}"
                        )

            print("OK: all phase state strings match controlled link count.")
            return 0
        finally:
            try:
                traci.close(False)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

