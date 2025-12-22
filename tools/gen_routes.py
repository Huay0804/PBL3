import argparse
import gzip
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PBL3_ROOT = os.path.normpath(os.path.join(THIS_DIR, ".."))

import sys

if PBL3_ROOT not in sys.path:
    sys.path.insert(0, PBL3_ROOT)

from pbl3_paper.sumo_lane_cells import load_experiment_config, resolve_config_paths, resolve_path  # noqa: E402


@dataclass(frozen=True)
class TurnMap:
    incoming_edges: List[str]
    straight_to: Dict[str, List[str]]
    left_to: Dict[str, List[str]]
    right_to: Dict[str, List[str]]
    uturn_to: Dict[str, List[str]]


def read_net_from_sumocfg(sumocfg: str) -> str:
    sumocfg = os.path.abspath(sumocfg)
    tree = ET.parse(sumocfg)
    root = tree.getroot()
    input_node = root.find("input")
    if input_node is None:
        raise RuntimeError("Invalid .sumocfg: missing <input>")
    net_el = input_node.find("net-file")
    if net_el is None or not (net_el.get("value") or "").strip():
        raise RuntimeError("Invalid .sumocfg: missing <net-file value=...>")
    return resolve_path(sumocfg, net_el.get("value").strip())


def _open_xml(path: str) -> ET.Element:
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            return ET.parse(handle).getroot()
    return ET.parse(path).getroot()


def build_turn_map_from_net(*, net_file: str, tls_id: str) -> TurnMap:
    root = _open_xml(net_file)

    moves: Dict[str, Dict[str, List[str]]] = {}
    for conn in root.findall("connection"):
        if (conn.get("tl") or "") != tls_id:
            continue
        from_edge = (conn.get("from") or "").strip()
        to_edge = (conn.get("to") or "").strip()
        dir_code = (conn.get("dir") or "").strip()
        if not from_edge or not to_edge or from_edge.startswith(":") or to_edge.startswith(":"):
            continue
        if dir_code not in {"r", "s", "l", "t"}:
            continue
        by_dir = moves.setdefault(from_edge, {})
        by_dir.setdefault(dir_code, [])
        if to_edge not in by_dir[dir_code]:
            by_dir[dir_code].append(to_edge)

    if not moves:
        raise RuntimeError(f"No <connection tl=...> found for tls_id={tls_id!r} in {net_file}")

    incoming_edges = sorted(moves.keys())
    straight_to: Dict[str, List[str]] = {}
    left_to: Dict[str, List[str]] = {}
    right_to: Dict[str, List[str]] = {}
    uturn_to: Dict[str, List[str]] = {}

    for inc in incoming_edges:
        by_dir = moves[inc]
        if "s" in by_dir and by_dir["s"]:
            straight_to[inc] = list(by_dir["s"])
        if "l" in by_dir and by_dir["l"]:
            left_to[inc] = list(by_dir["l"])
        if "r" in by_dir and by_dir["r"]:
            right_to[inc] = list(by_dir["r"])
        if "t" in by_dir and by_dir["t"]:
            uturn_to[inc] = list(by_dir["t"])

    if not straight_to:
        raise RuntimeError(
            f"Could not find any straight movements (dir='s') for tls_id={tls_id!r}. "
            "Check the net.xml <connection dir=...> attributes."
        )

    return TurnMap(
        incoming_edges=incoming_edges,
        straight_to=straight_to,
        left_to=left_to,
        right_to=right_to,
        uturn_to=uturn_to,
    )


def _weibull_depart_times(*, rng: np.random.RandomState, n: int, end: int, shape: float) -> np.ndarray:
    timings = rng.weibull(shape, n)
    timings = np.sort(timings)
    if len(timings) < 2:
        return np.array([0], dtype=int)

    min_old = math.floor(float(timings[1]))
    max_old = math.ceil(float(timings[-1]))
    if max_old <= min_old:
        return np.zeros(n, dtype=int)

    min_new = 0.0
    max_new = float(end)
    scaled = ((max_new - min_new) / (max_old - min_old)) * (timings - min_old) + min_new
    depart = np.rint(scaled).astype(int)
    depart = np.clip(depart, 0, int(end))
    return depart


def _normalize_ratios(straight: float, turn: float, uturn: float) -> List[float]:
    total = float(straight) + float(turn) + float(uturn)
    if total <= 0:
        return [1.0, 0.0, 0.0]
    return [float(straight) / total, float(turn) / total, float(uturn) / total]


def _pick_from_list(rng: np.random.RandomState, values: List[str]) -> str:
    if not values:
        return ""
    return str(rng.choice(values))


def generate_routes_file(
    *,
    out_route_file: str,
    seed: int,
    turns: TurnMap,
    n_vehicles: int,
    end: int,
    weibull_shape: float,
    straight_ratio: float,
    turn_ratio: float,
    uturn_ratio: float,
    allow_uturn: bool,
    depart_lane: str = "best",
    depart_speed: str = "5",
    vehicle_prefix: str = "veh",
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_route_file)), exist_ok=True)

    rng = np.random.RandomState(int(seed))
    departs = _weibull_depart_times(rng=rng, n=int(n_vehicles), end=int(end), shape=float(weibull_shape))

    incoming = list(turns.incoming_edges)
    if not incoming:
        raise RuntimeError("No incoming edges found to generate routes.")

    straight_ratio, turn_ratio, uturn_ratio = _normalize_ratios(straight_ratio, turn_ratio, uturn_ratio)
    if not allow_uturn:
        uturn_ratio = 0.0
        straight_ratio, turn_ratio, _ = _normalize_ratios(straight_ratio, turn_ratio, 0.0)

    depart_speed = str(depart_speed).strip()
    include_depart_speed = depart_speed.lower() not in {"", "auto", "default", "none"}

    with open(out_route_file, "w", encoding="utf-8", newline="\n") as f:
        f.write("<routes>\n")
        f.write(
            '  <vType id="standard_car" accel="1.0" decel="4.5" length="5.0" minGap="2.5" '
            'maxSpeed="25" sigma="0.5"/>\n'
        )

        for idx, depart in enumerate(departs.tolist()):
            from_edge = str(rng.choice(incoming))
            r = float(rng.uniform())

            to_edge = ""
            if r < straight_ratio and from_edge in turns.straight_to:
                to_edge = _pick_from_list(rng, turns.straight_to.get(from_edge, []))
            elif r < straight_ratio + turn_ratio:
                candidates = []
                candidates.extend(turns.left_to.get(from_edge, []))
                candidates.extend(turns.right_to.get(from_edge, []))
                if candidates:
                    to_edge = _pick_from_list(rng, candidates)
            else:
                if allow_uturn:
                    to_edge = _pick_from_list(rng, turns.uturn_to.get(from_edge, []))

            if not to_edge:
                to_edge = _pick_from_list(rng, turns.straight_to.get(from_edge, []))
            if not to_edge:
                to_edge = _pick_from_list(rng, turns.left_to.get(from_edge, []))
            if not to_edge:
                to_edge = _pick_from_list(rng, turns.right_to.get(from_edge, []))
            if not to_edge:
                to_edge = _pick_from_list(rng, turns.uturn_to.get(from_edge, []))
            if not to_edge:
                continue

            veh_id = f"{vehicle_prefix}_{seed}_{idx}"
            if include_depart_speed:
                f.write(
                    f'  <vehicle id="{veh_id}" type="standard_car" depart="{int(depart)}" '
                    f'departLane="{depart_lane}" departSpeed="{depart_speed}">\n'
                )
            else:
                f.write(
                    f'  <vehicle id="{veh_id}" type="standard_car" depart="{int(depart)}" '
                    f'departLane="{depart_lane}">\n'
                )
            f.write(f'    <route edges="{from_edge} {to_edge}"/>\n')
            f.write("  </vehicle>\n")

        f.write("</routes>\n")

    return os.path.abspath(out_route_file)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate per-seed route files using experiment_config.yaml.")
    p.add_argument("--config", default=os.path.join(PBL3_ROOT, "experiment_config.yaml"))
    p.add_argument("--seeds", type=int, nargs="+", required=True, help="Seeds to generate (e.g., 0 1 2 ... 19)")
    p.add_argument("--outdir", default="", help="Output directory for routes_seed{seed}.rou.xml")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_experiment_config(args.config)
    paths = resolve_config_paths(args.config, config)
    sumocfg = paths["sumocfg"]
    tls_id = paths["tls_id"]

    exp = config.get("experiment", {})
    traffic = config.get("traffic", {})
    paths_cfg = config.get("paths", {})

    vehicles = int(exp.get("vehicles", 1000))
    end = int(exp.get("sim_seconds", 5400))
    weibull_shape = float(traffic.get("weibull_shape", 2.0))
    straight_ratio = float(traffic.get("straight_ratio", 0.75))
    turn_ratio = float(traffic.get("turn_ratio", 0.25))
    uturn_ratio = float(traffic.get("uturn_ratio", 0.0))
    allow_uturn = bool(traffic.get("allow_uturn", False))
    depart_lane = str(traffic.get("depart_lane", "best"))
    depart_speed = str(traffic.get("depart_speed", "5"))

    net_file = read_net_from_sumocfg(sumocfg)
    turns = build_turn_map_from_net(net_file=net_file, tls_id=tls_id)

    default_outdir = str(paths_cfg.get("routes_dir", "results/routes"))
    outdir = args.outdir.strip() or default_outdir
    outdir = resolve_path(args.config, outdir)

    os.makedirs(outdir, exist_ok=True)
    print(f"net: {net_file}")
    print(f"tls: {tls_id}")
    print(f"incoming edges ({len(turns.incoming_edges)}): {', '.join(turns.incoming_edges)}")
    print(f"outdir: {outdir}")

    for seed in args.seeds:
        out_path = os.path.join(outdir, f"routes_seed{int(seed)}.rou.xml")
        generate_routes_file(
            out_route_file=out_path,
            seed=int(seed),
            turns=turns,
            n_vehicles=int(vehicles),
            end=int(end),
            weibull_shape=float(weibull_shape),
            straight_ratio=float(straight_ratio),
            turn_ratio=float(turn_ratio),
            uturn_ratio=float(uturn_ratio),
            allow_uturn=bool(allow_uturn),
            depart_lane=str(depart_lane),
            depart_speed=str(depart_speed),
            vehicle_prefix="veh",
        )
        print(f"Wrote: {os.path.abspath(out_path)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
