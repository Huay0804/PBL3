import argparse
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from typing import List, Optional, Sequence


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


def read_net_from_sumocfg(sumocfg: str) -> str:
    sumocfg = os.path.abspath(sumocfg)
    tree = ET.parse(sumocfg)
    root = tree.getroot()
    input_node = root.find("input")
    if input_node is None:
        raise RuntimeError("Invalid .sumocfg: missing <input>")
    net_el = input_node.find("net-file")
    if net_el is None or not net_el.get("value"):
        raise RuntimeError("Invalid .sumocfg: missing <net-file value=...>")
    net_rel = net_el.get("value").strip()
    if os.path.isabs(net_rel):
        return net_rel
    return os.path.normpath(os.path.join(os.path.dirname(sumocfg), net_rel))


@contextmanager
def pushd(path: str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def run_random_trips(
    *,
    net_file: str,
    out_trips: str,
    out_routes: str,
    seed: int,
    end: int,
    insertion_density: float,
    period: Optional[float],
    insertion_rate: Optional[float],
    fringe_factor: float,
    min_distance: float,
    vclass: str,
    prefix: str,
    validate: bool,
    remove_loops: bool,
    lanes: bool,
    trip_attributes: str,
    fringe_start_attributes: str,
    extra_args: Optional[Sequence[str]] = None,
) -> None:
    sumo_home = ensure_sumo_tools()
    random_trips = os.path.join(sumo_home, "tools", "randomTrips.py")
    if not os.path.isfile(random_trips):
        raise RuntimeError(f"randomTrips.py not found at {random_trips}")

    os.makedirs(os.path.dirname(os.path.abspath(out_trips)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_routes)), exist_ok=True)

    cmd: List[str] = [
        sys.executable,
        random_trips,
        "-n",
        os.path.abspath(net_file),
        "-o",
        os.path.abspath(out_trips),
        "-r",
        os.path.abspath(out_routes),
        "-b",
        "0",
        "-e",
        str(int(end)),
        "--seed",
        str(int(seed)),
    ]

    if period is not None:
        cmd.extend(["-p", str(float(period))])
    elif insertion_rate is not None:
        cmd.extend(["--insertion-rate", str(float(insertion_rate))])
    else:
        cmd.extend(["--insertion-density", str(float(insertion_density))])

    cmd.extend(
        [
            "--fringe-factor",
            str(float(fringe_factor)),
            "--min-distance",
            str(float(min_distance)),
            "--vehicle-class",
            vclass,
            "--prefix",
            prefix,
            "--trip-attributes",
            trip_attributes,
            "--fringe-start-attributes",
            fringe_start_attributes,
        ]
    )

    if lanes:
        cmd.append("--lanes")
    if validate:
        cmd.append("--validate")
    if remove_loops:
        cmd.append("--remove-loops")
    if extra_args:
        cmd.extend(list(extra_args))

    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate per-seed SUMO route files using randomTrips.py (OSMWebWizard-style).")
    parser.add_argument("--sumocfg", required=True, help="Path to .sumocfg (used to locate net-file)")
    parser.add_argument("--out-dir", default="generated_routes", help="Output directory (relative to sumocfg dir)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)), help="Seeds to generate (default: 0..19)")
    parser.add_argument("--end", type=int, default=6000, help="End time (seconds)")

    demand = parser.add_mutually_exclusive_group()
    demand.add_argument("--period", type=float, default=None, help="randomTrips period (-p)")
    demand.add_argument("--insertion-rate", type=float, default=None, help="randomTrips --insertion-rate")
    demand.add_argument("--insertion-density", type=float, default=12.0, help="randomTrips --insertion-density (default: 12)")
    parser.add_argument("--fringe-factor", type=float, default=5.0)
    parser.add_argument("--min-distance", type=float, default=300.0)

    parser.add_argument("--vclass", default="passenger")
    parser.add_argument("--prefix", default="veh")
    parser.add_argument("--trip-attributes", default='departLane="best"', help='Passed to randomTrips.py --trip-attributes')
    parser.add_argument(
        "--fringe-start-attributes",
        default='departSpeed="max"',
        help='Passed to randomTrips.py --fringe-start-attributes',
    )

    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--no-remove-loops", action="store_true")
    parser.add_argument("--no-lanes", action="store_true")
    args, extra = parser.parse_known_args()

    demand_tokens = {"-p", "--period", "--insertion-rate", "--insertion-density"}
    if any(token in demand_tokens for token in extra):
        raise RuntimeError(
            "Do not pass demand options to randomTrips.py via extra args. Use gen_routes.py --period/--insertion-rate/--insertion-density."
        )

    sumocfg = os.path.abspath(args.sumocfg)
    if not os.path.isfile(sumocfg):
        raise RuntimeError(f"sumocfg not found: {sumocfg}")

    net_file = read_net_from_sumocfg(sumocfg)
    if not os.path.isfile(net_file):
        raise RuntimeError(f"net file not found: {net_file}")

    scenario_dir = os.path.dirname(sumocfg)
    out_dir = os.path.join(scenario_dir, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"sumocfg: {sumocfg}")
    print(f"net:     {net_file}")
    print(f"out_dir: {out_dir}")
    print(f"seeds:   {args.seeds}")

    with pushd(scenario_dir):
        for seed in args.seeds:
            trips = os.path.join(out_dir, f"trips_seed{seed}.trips.xml")
            routes = os.path.join(out_dir, f"routes_seed{seed}.rou.xml")
            run_random_trips(
                net_file=net_file,
                out_trips=trips,
                out_routes=routes,
                seed=int(seed),
                end=int(args.end),
                insertion_density=float(args.insertion_density),
                period=None if args.period is None else float(args.period),
                insertion_rate=None if args.insertion_rate is None else float(args.insertion_rate),
                fringe_factor=float(args.fringe_factor),
                min_distance=float(args.min_distance),
                vclass=str(args.vclass),
                prefix=str(args.prefix),
                validate=not bool(args.no_validate),
                remove_loops=not bool(args.no_remove_loops),
                lanes=not bool(args.no_lanes),
                trip_attributes=str(args.trip_attributes),
                fringe_start_attributes=str(args.fringe_start_attributes),
                extra_args=extra,
            )
            print(f"generated: {routes}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
