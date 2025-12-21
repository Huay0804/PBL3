import argparse
import csv
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


def lane_to_edge(lane_id: str) -> str:
    return lane_id.rsplit("_", 1)[0]


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


def infer_outgoing_edges(tls_id: str, traci_module) -> List[str]:
    controlled_links = traci_module.trafficlight.getControlledLinks(tls_id)
    edges: Set[str] = set()
    for link in controlled_links:
        for conn in link:
            if not conn or len(conn) < 2:
                continue
            to_lane = conn[1]
            if to_lane and not to_lane.startswith(":"):
                edges.add(lane_to_edge(to_lane))
    return sorted(edges)


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


def mean_speed_in(lanes: Sequence[str], traci_module) -> float:
    weighted = 0.0
    total = 0.0
    for lane in lanes:
        count = float(traci_module.lane.getLastStepVehicleNumber(lane))
        speed = float(traci_module.lane.getLastStepMeanSpeed(lane))
        weighted += speed * count
        total += count
    if total <= 0:
        return 0.0
    return float(weighted / total)


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline fixed-time logging (no TLS overrides).")
    parser.add_argument("--sumocfg", required=True, help="Path to .sumocfg")
    parser.add_argument("--tls-id", required=True, help="Traffic light id")
    parser.add_argument("--seed", type=int, default=0, help="SUMO seed")
    parser.add_argument("--duration", type=int, default=6000, help="Max simulation seconds to run")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--gui", type=int, default=0, help="1 to use sumo-gui")
    parser.add_argument("--log-every", type=int, default=5, help="Log every N seconds (default: 5)")
    parser.add_argument(
        "--route-files",
        default=None,
        help="Override route files (comma-separated). If omitted, uses the .sumocfg route-files as-is.",
    )
    args = parser.parse_args()

    sumocfg = os.path.abspath(args.sumocfg)
    if not os.path.isfile(sumocfg):
        raise RuntimeError(f"sumocfg not found: {sumocfg}")

    out_csv = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    sumo_bin = resolve_sumo_binary(gui=bool(args.gui))

    ensure_sumo_tools()
    import traci  # noqa: E402

    scenario_dir = os.path.dirname(sumocfg)

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

    with pushd(scenario_dir):
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

            incoming_edges = sorted({lane_to_edge(lane) for lane in lanes})
            outgoing_edges = infer_outgoing_edges(args.tls_id, traci)
            outgoing_edges_set = set(outgoing_edges)
            if not outgoing_edges:
                raise RuntimeError(f"Could not infer outgoing edges for tls-id={args.tls_id!r}")

            seen_incoming: Set[str] = set()
            seen_throughput: Set[str] = set()
            throughput_junction = 0

            arrived_total = 0
            departed_total = 0

            with open(out_csv, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "time",
                        "sum_queue",
                        "sum_wait",
                        "mean_speed_in",
                        "throughput_junction",
                        "departed",
                        "arrived",
                        "running",
                        "departed_total",
                        "arrived_total",
                    ]
                )

                while True:
                    traci.simulationStep()
                    t = int(traci.simulation.getTime())
                    departed = int(traci.simulation.getDepartedNumber())
                    arrived = int(traci.simulation.getArrivedNumber())
                    running = int(traci.simulation.getMinExpectedNumber())
                    departed_total += departed
                    arrived_total += arrived

                    # Junction throughput: count vehicles entering any outgoing edge for the first time.
                    for veh_id in traci.vehicle.getIDList():
                        try:
                            road_id = traci.vehicle.getRoadID(veh_id)
                        except Exception:
                            continue
                        if road_id in incoming_edges:
                            seen_incoming.add(veh_id)
                        if road_id in outgoing_edges_set and veh_id in seen_incoming and veh_id not in seen_throughput:
                            seen_throughput.add(veh_id)
                            throughput_junction += 1

                    if (t % int(args.log_every)) == 0:
                        q = sum_queue(lanes, traci)
                        w = sum_wait(lanes_set, traci)
                        ms = mean_speed_in(lanes, traci)
                        writer.writerow(
                            [
                                t,
                                f"{q:.0f}",
                                f"{w:.1f}",
                                f"{ms:.3f}",
                                throughput_junction,
                                departed,
                                arrived,
                                running,
                                departed_total,
                                arrived_total,
                            ]
                        )

                    if t >= int(args.duration):
                        break
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        break

            print(f"Wrote: {out_csv}")
            print(
                "Throughput method: count unique vehicles that enter any outgoing edge of the junction for the first time "
                "(detected by vehicle.roadID in outgoing_edges inferred from controlledLinks)."
            )
            print("arrived/arrived_total are kept as debug fields (vehicles reaching destination).")
            print(f"incoming_edges: {', '.join(incoming_edges)}")
            print(f"outgoing_edges: {', '.join(outgoing_edges)}")
            if route_files:
                print("route_files_override:")
                for rf in route_files:
                    print(f"  - {rf}")
            return 0
        finally:
            try:
                traci.close(False)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
