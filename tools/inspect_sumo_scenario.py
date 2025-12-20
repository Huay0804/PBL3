import argparse
import gzip
import os
import sys
import xml.etree.ElementTree as ET


def _open_text(path: str):
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def _resolve_from(base_file: str, maybe_relative: str) -> str:
    if os.path.isabs(maybe_relative):
        return maybe_relative
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(base_file)), maybe_relative))


def read_sumocfg(sumocfg_path: str) -> dict:
    with _open_text(sumocfg_path) as handle:
        tree = ET.parse(handle)
    root = tree.getroot()

    input_node = root.find("input")
    if input_node is None:
        raise RuntimeError("Invalid .sumocfg: missing <input> section")

    net_file = input_node.findtext("net-file[@value]", default=None)
    if not net_file:
        net_el = input_node.find("net-file")
        net_file = net_el.get("value") if net_el is not None else None
    if not net_file:
        raise RuntimeError("Invalid .sumocfg: missing <net-file value=...>")

    route_files = ""
    route_el = input_node.find("route-files")
    if route_el is not None:
        route_files = route_el.get("value", "")

    additional_files = ""
    add_el = input_node.find("additional-files")
    if add_el is not None:
        additional_files = add_el.get("value", "")

    net_path = _resolve_from(sumocfg_path, net_file)
    route_paths = [
        _resolve_from(sumocfg_path, p.strip())
        for p in route_files.split(",")
        if p.strip()
    ]
    additional_paths = [
        _resolve_from(sumocfg_path, p.strip())
        for p in additional_files.split(",")
        if p.strip()
    ]

    return {
        "sumocfg": os.path.abspath(sumocfg_path),
        "net": net_path,
        "routes": route_paths,
        "additional": additional_paths,
    }


def parse_net_tls(net_path: str) -> dict:
    with _open_text(net_path) as handle:
        tree = ET.parse(handle)
    root = tree.getroot()

    tls = {}
    for tl in root.findall("tlLogic"):
        tls_id = tl.get("id")
        if not tls_id:
            continue
        phases = [p.get("state", "") for p in tl.findall("phase")]
        tls[tls_id] = phases
    return tls


def infer_controlled_lanes(net_path: str, tls_id: str) -> list:
    with _open_text(net_path) as handle:
        tree = ET.parse(handle)
    root = tree.getroot()

    lanes = set()
    for conn in root.findall("connection"):
        if conn.get("tl") != tls_id:
            continue
        from_edge = conn.get("from")
        from_lane = conn.get("fromLane")
        if not from_edge or from_edge.startswith(":") or from_lane is None:
            continue
        lanes.add(f"{from_edge}_{from_lane}")
    return sorted(lanes)


def classify_phases(phases: list) -> dict:
    greens = []
    yellows = []
    others = []
    for idx, state in enumerate(phases):
        s = state or ""
        if "y" in s or "Y" in s:
            yellows.append(idx)
        elif ("G" in s) or ("g" in s):
            greens.append(idx)
        else:
            others.append(idx)
    return {"green": greens, "yellow": yellows, "other": others}


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description="Inspect a SUMO .sumocfg scenario (no SUMO install required).")
    parser.add_argument("--sumocfg", required=True, help="Path to .sumocfg")
    parser.add_argument("--tls-id", default=None, help="Traffic light id to inspect (tlLogic id)")
    args = parser.parse_args(argv)

    cfg = read_sumocfg(args.sumocfg)
    if not os.path.isfile(cfg["net"]):
        raise RuntimeError(f"Net file not found: {cfg['net']}")

    tls = parse_net_tls(cfg["net"])
    if not tls:
        print("No <tlLogic> found in net file.")
        return 0

    print(f"sumocfg: {cfg['sumocfg']}")
    print(f"net:     {cfg['net']}")
    if cfg["routes"]:
        print("routes:")
        for p in cfg["routes"]:
            print(f"  - {p}")

    print("traffic lights:")
    for tls_id, phases in tls.items():
        cls = classify_phases(phases)
        print(
            f"  - {tls_id}: phases={len(phases)} green={cls['green']} yellow={cls['yellow']}"
        )

    tls_id = args.tls_id or next(iter(tls.keys()))
    if tls_id not in tls:
        raise RuntimeError(f"tls-id {tls_id!r} not found. Available: {', '.join(tls.keys())}")

    lanes = infer_controlled_lanes(cfg["net"], tls_id)
    edges = sorted({lane.rsplit("_", 1)[0] for lane in lanes})
    print(f"inspect tls-id: {tls_id}")
    print(f"controlled lanes ({len(lanes)}):")
    for lane in lanes:
        print(f"  - {lane}")
    print(f"incoming edges ({len(edges)}):")
    for edge in edges:
        print(f"  - {edge}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

