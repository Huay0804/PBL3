import argparse
import os
import shutil
import sys
from contextlib import contextmanager
from typing import Iterable, List, Optional, Sequence, Set, Tuple


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
    if lanes:
        return sorted(lanes)
    return unique_sorted(
        [l for l in traci_module.trafficlight.getControlledLanes(tls_id) if l and not l.startswith(":")]
    )


def sum_queue(lanes: Sequence[str], traci_module) -> float:
    return float(sum(traci_module.lane.getLastStepHaltingNumber(lane) for lane in lanes))


def sum_wait(lanes_set: Set[str], traci_module) -> float:
    total = 0.0
    for veh_id in traci_module.vehicle.getIDList():
        try:
            lane_id = traci_module.vehicle.getLaneID(veh_id)
        except Exception:
            continue
        if lane_id in lanes_set:
            try:
                total += float(traci_module.vehicle.getAccumulatedWaitingTime(veh_id))
            except Exception:
                continue
    return float(total)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test: force TLS phases and log queue/wait via TraCI.")
    parser.add_argument("--sumocfg", required=True, help="Path to .sumocfg")
    parser.add_argument("--tls-id", required=True, help="Traffic light id")
    parser.add_argument("--gui", type=int, default=0, help="1 to use sumo-gui")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1, help="Log every N seconds (default: 1)")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup seconds before forced-phase schedule")
    parser.add_argument(
        "--route-files",
        default=None,
        help="Override route files (comma-separated). If omitted, uses the .sumocfg route-files as-is.",
    )
    args = parser.parse_args()

    sumocfg = os.path.abspath(args.sumocfg)
    if not os.path.isfile(sumocfg):
        raise RuntimeError(f"sumocfg not found: {sumocfg}")

    sumo_bin = resolve_sumo_binary(gui=bool(args.gui))

    ensure_sumo_tools()
    import traci  # noqa: E402

    scenario_dir = os.path.dirname(sumocfg)
    schedule = [(0, 10), (2, 10), (0, 10), (2, 10)]

    with pushd(scenario_dir):
        route_files: Optional[List[str]] = None
        if args.route_files:
            route_files = []
            for part in str(args.route_files).split(","):
                part = part.strip()
                if not part:
                    continue
                path = os.path.join(scenario_dir, part) if not os.path.isabs(part) else part
                route_files.append(os.path.abspath(path))
            if not route_files:
                route_files = None
            else:
                for rf in route_files:
                    if not os.path.isfile(rf):
                        raise RuntimeError(f"route file not found: {rf}")

        cmd = [
            sumo_bin,
            "-c",
            sumocfg,
            "--seed",
            str(int(args.seed)),
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
        ]
        if route_files:
            cmd.extend(["--route-files", ",".join(route_files)])

        traci.start(cmd)
        try:
            tls_ids = set(traci.trafficlight.getIDList())
            if args.tls_id not in tls_ids:
                raise RuntimeError(f"tls-id {args.tls_id!r} not found. Available: {', '.join(sorted(tls_ids))}")

            lanes = infer_incoming_lanes(args.tls_id, traci)
            if not lanes:
                raise RuntimeError(f"No incoming lanes found for tls-id={args.tls_id!r}")
            lanes_set = set(lanes)
            print(f"incoming_lanes({len(lanes)}): {', '.join(lanes)}")
            if route_files:
                print("route_files_override:")
                for rf in route_files:
                    print(f"  - {rf}")

            if int(args.warmup) > 0:
                for _ in range(int(args.warmup)):
                    traci.simulationStep()
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        break
                print(f"warmup_done: t={int(traci.simulation.getTime())}")

            print("time,phase,sum_queue,sum_wait")

            print("segment,phase,t_start,t_end,queue_before,queue_after,wait_before,wait_after,delta_queue,delta_wait")

            segment_idx = 0
            for phase_index, seconds in schedule:
                segment_idx += 1
                t_start = int(traci.simulation.getTime())
                q_before = sum_queue(lanes, traci)
                w_before = sum_wait(lanes_set, traci)

                traci.trafficlight.setPhase(args.tls_id, int(phase_index))
                traci.trafficlight.setPhaseDuration(args.tls_id, float(seconds))

                for _ in range(int(seconds)):
                    traci.simulationStep()
                    t = int(traci.simulation.getTime())
                    if (t % int(args.log_every)) == 0:
                        p = int(traci.trafficlight.getPhase(args.tls_id))
                        q = sum_queue(lanes, traci)
                        w = sum_wait(lanes_set, traci)
                        print(f"{t},{p},{q:.0f},{w:.1f}")

                    if traci.simulation.getMinExpectedNumber() <= 0:
                        break

                t_end = int(traci.simulation.getTime())
                q_after = sum_queue(lanes, traci)
                w_after = sum_wait(lanes_set, traci)
                print(
                    f"{segment_idx},{phase_index},{t_start},{t_end},{q_before:.0f},{q_after:.0f},{w_before:.1f},{w_after:.1f},{(q_after-q_before):.0f},{(w_after-w_before):.1f}"
                )

                if traci.simulation.getMinExpectedNumber() <= 0:
                    break

            return 0
        finally:
            try:
                traci.close(False)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
