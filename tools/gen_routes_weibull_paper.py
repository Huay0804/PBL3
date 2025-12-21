import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TurnMap:
    incoming_edges: List[str]
    straight_to: Dict[str, str]
    left_to: Dict[str, str]
    right_to: Dict[str, str]


def _resolve_from(base_file: str, maybe_relative: str) -> str:
    if os.path.isabs(maybe_relative):
        return maybe_relative
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(base_file)), maybe_relative))


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

    return _resolve_from(sumocfg, net_el.get("value").strip())


def build_turn_map_from_net(*, net_file: str, tls_id: str) -> TurnMap:
    tree = ET.parse(net_file)
    root = tree.getroot()

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
    straight_to: Dict[str, str] = {}
    left_to: Dict[str, str] = {}
    right_to: Dict[str, str] = {}

    for inc in incoming_edges:
        by_dir = moves[inc]
        if "s" in by_dir and by_dir["s"]:
            straight_to[inc] = by_dir["s"][0]
        if "l" in by_dir and by_dir["l"]:
            left_to[inc] = by_dir["l"][0]
        if "r" in by_dir and by_dir["r"]:
            right_to[inc] = by_dir["r"][0]

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


def generate_routes_file(
    *,
    out_route_file: str,
    seed: int,
    turns: TurnMap,
    n_vehicles: int,
    end: int,
    weibull_shape: float = 2.0,
    straight_prob: float = 0.75,
    depart_speed: str = "10",
    vehicle_prefix: str = "veh",
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_route_file)), exist_ok=True)

    rng = np.random.RandomState(int(seed))
    departs = _weibull_depart_times(rng=rng, n=int(n_vehicles), end=int(end), shape=float(weibull_shape))

    incoming = list(turns.incoming_edges)
    if not incoming:
        raise RuntimeError("No incoming edges found to generate routes.")

    def _pick_turn_to(from_edge: str) -> str:
        candidates: List[str] = []
        if from_edge in turns.left_to:
            candidates.append(turns.left_to[from_edge])
        if from_edge in turns.right_to:
            candidates.append(turns.right_to[from_edge])
        if not candidates:
            return turns.straight_to.get(from_edge) or turns.straight_to[next(iter(turns.straight_to.keys()))]
        return str(rng.choice(candidates))

    with open(out_route_file, "w", encoding="utf-8", newline="\n") as f:
        f.write("<routes>\n")
        f.write(
            '  <vType id="standard_car" accel="1.0" decel="4.5" length="5.0" minGap="2.5" '
            'maxSpeed="25" sigma="0.5"/>\n'
        )

        for idx, depart in enumerate(departs.tolist()):
            from_edge = str(rng.choice(incoming))
            if rng.uniform() < float(straight_prob) and from_edge in turns.straight_to:
                to_edge = turns.straight_to[from_edge]
            else:
                to_edge = _pick_turn_to(from_edge)
            veh_id = f"{vehicle_prefix}_{seed}_{idx}"
            f.write(
                f'  <vehicle id="{veh_id}" type="standard_car" depart="{int(depart)}" '
                f'departLane="random" departSpeed="{depart_speed}">\n'
            )
            f.write(f'    <route edges="{from_edge} {to_edge}"/>\n')
            f.write("  </vehicle>\n")

        f.write("</routes>\n")

    return os.path.abspath(out_route_file)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate per-seed route files using Weibull demand (paper-like: 1000 veh, 5400s, 75% straight)."
    )
    p.add_argument("--sumocfg", required=True, help="Path to .sumocfg (used to locate .net.xml)")
    p.add_argument("--tls-id", required=True, help="Traffic light system id (e.g., GS_420249146)")
    p.add_argument("--outdir", required=True, help="Output directory for routes_seed{seed}.rou.xml")
    p.add_argument("--seeds", type=int, nargs="+", required=True, help="Seeds to generate (e.g., 0 1 2 ... 19)")

    p.add_argument("--vehicles", type=int, default=1000, help="Number of vehicles per route file (default: 1000)")
    p.add_argument("--end", type=int, default=5400, help="Simulation end time in seconds (default: 5400)")
    p.add_argument("--weibull-shape", type=float, default=2.0, help="Weibull shape parameter k (default: 2.0)")
    p.add_argument("--straight-prob", type=float, default=0.75, help="Probability a vehicle goes straight (default: 0.75)")
    p.add_argument("--depart-speed", default="10", help='departSpeed attribute (default: "10")')
    p.add_argument("--prefix", default="veh", help="Vehicle id prefix (default: veh)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    net_file = read_net_from_sumocfg(args.sumocfg)
    turns = build_turn_map_from_net(net_file=net_file, tls_id=args.tls_id)

    os.makedirs(args.outdir, exist_ok=True)
    print(f"net: {net_file}")
    print(f"tls: {args.tls_id}")
    print(f"incoming edges ({len(turns.incoming_edges)}): {', '.join(turns.incoming_edges)}")

    for seed in args.seeds:
        out_path = os.path.join(args.outdir, f"routes_seed{int(seed)}.rou.xml")
        generate_routes_file(
            out_route_file=out_path,
            seed=int(seed),
            turns=turns,
            n_vehicles=int(args.vehicles),
            end=int(args.end),
            weibull_shape=float(args.weibull_shape),
            straight_prob=float(args.straight_prob),
            depart_speed=str(args.depart_speed),
            vehicle_prefix=str(args.prefix),
        )
        print(f"Wrote: {os.path.abspath(out_path)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

