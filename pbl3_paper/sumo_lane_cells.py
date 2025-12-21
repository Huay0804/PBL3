import gzip
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


try:
    import yaml
except Exception:  # pragma: no cover - handled at runtime
    yaml = None


@dataclass(frozen=True)
class TLSConn:
    link_index: int
    from_edge: str
    from_lane: str
    to_edge: str
    dir_code: str  # r / s / l / t


@dataclass(frozen=True)
class TLSProgram:
    phase_states: List[str]
    phase_durations: List[float]


@dataclass(frozen=True)
class LaneGroups:
    tls_id: str
    net_file: str
    incoming_edges: List[str]
    outgoing_edges: List[str]
    arm_order: List[str]
    edge_to_arm: Dict[str, str]
    groups: Dict[str, Dict[str, List[str]]]
    group_order: List[Tuple[str, str]]
    lane_to_group: Dict[str, str]
    conns: List[TLSConn]
    program: TLSProgram


@dataclass(frozen=True)
class PhaseSemanticsReport:
    ok: bool
    issues: List[str]
    warnings: List[str]
    details: Dict[int, Dict[str, object]]


def _open_xml(path: str) -> ET.Element:
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            return ET.parse(handle).getroot()
    return ET.parse(path).getroot()


def load_experiment_config(config_path: str) -> Dict[str, object]:
    if yaml is None:
        raise RuntimeError("Missing PyYAML. Install with: pip install pyyaml")
    config_path = os.path.abspath(config_path)
    if not os.path.isfile(config_path):
        raise RuntimeError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise RuntimeError("Invalid experiment_config.yaml format.")
    return data


def resolve_path(base_file: str, maybe_rel: str) -> str:
    if os.path.isabs(maybe_rel):
        return maybe_rel
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(base_file)), maybe_rel))


def resolve_config_paths(config_path: str, config: Dict[str, object]) -> Dict[str, str]:
    project = config.get("project", {})
    if not isinstance(project, dict):
        raise RuntimeError("Invalid config: project must be a mapping")
    sumocfg = str(project.get("sumocfg") or "").strip()
    if not sumocfg:
        raise RuntimeError("Config missing project.sumocfg")
    tls_id = str(project.get("tls_id") or "").strip()
    if not tls_id:
        raise RuntimeError("Config missing project.tls_id")
    scenario_dir = str(project.get("scenario_dir") or "").strip()
    if not scenario_dir:
        scenario_dir = os.path.dirname(sumocfg)
    sumocfg_abs = resolve_path(config_path, sumocfg)
    scenario_abs = resolve_path(config_path, scenario_dir)
    return {"sumocfg": sumocfg_abs, "scenario_dir": scenario_abs, "tls_id": tls_id}


def read_sumocfg(sumocfg_path: str) -> Dict[str, object]:
    sumocfg_path = os.path.abspath(sumocfg_path)
    root = _open_xml(sumocfg_path)
    input_node = root.find("input")
    if input_node is None:
        raise RuntimeError("Invalid .sumocfg: missing <input>")

    def _val(tag: str) -> str:
        el = input_node.find(tag)
        return (el.get("value") or "").strip() if el is not None else ""

    net_rel = _val("net-file")
    if not net_rel:
        raise RuntimeError("Invalid .sumocfg: missing <net-file value=...>")

    def _resolve(base: str, maybe_rel: str) -> str:
        if os.path.isabs(maybe_rel):
            return maybe_rel
        return os.path.normpath(os.path.join(os.path.dirname(base), maybe_rel))

    net_path = _resolve(sumocfg_path, net_rel)
    routes = [_resolve(sumocfg_path, p.strip()) for p in _val("route-files").split(",") if p.strip()]
    additional = [_resolve(sumocfg_path, p.strip()) for p in _val("additional-files").split(",") if p.strip()]
    return {"sumocfg": sumocfg_path, "net": net_path, "routes": routes, "additional": additional}


def _parse_tls_program(root: ET.Element, tls_id: str) -> TLSProgram:
    tl = root.find(f".//tlLogic[@id='{tls_id}']")
    if tl is None:
        raise RuntimeError(f"TLS program not found for tls_id={tls_id}")
    phase_states: List[str] = []
    phase_durations: List[float] = []
    for phase in tl.findall("phase"):
        phase_states.append(str(phase.get("state") or ""))
        phase_durations.append(float(phase.get("duration") or 0.0))
    return TLSProgram(phase_states=phase_states, phase_durations=phase_durations)


def _parse_net_tls(
    *, net_file: str, tls_id: str
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Dict[str, object]], Dict[str, Set[str]], List[TLSConn], TLSProgram]:
    root = _open_xml(net_file)

    nodes: Dict[str, Tuple[float, float]] = {}
    for junc in root.findall("junction"):
        node_id = (junc.get("id") or "").strip()
        if not node_id:
            continue
        try:
            x = float(junc.get("x") or 0.0)
            y = float(junc.get("y") or 0.0)
        except Exception:
            x, y = 0.0, 0.0
        nodes[node_id] = (x, y)

    edges: Dict[str, Dict[str, object]] = {}
    for edge in root.findall("edge"):
        edge_id = (edge.get("id") or "").strip()
        if not edge_id or edge_id.startswith(":"):
            continue
        if (edge.get("function") or "").strip() == "internal":
            continue
        from_node = (edge.get("from") or "").strip()
        to_node = (edge.get("to") or "").strip()
        lanes: List[Tuple[int, str, float]] = []
        for lane in edge.findall("lane"):
            lane_id = (lane.get("id") or "").strip()
            if not lane_id:
                continue
            idx = int(lane.get("index") or 0)
            length = float(lane.get("length") or 0.0)
            lanes.append((idx, lane_id, length))
        lanes.sort(key=lambda x: x[0])
        edges[edge_id] = {"from": from_node, "to": to_node, "lanes": lanes}

    lane_dirs: Dict[str, Set[str]] = {}
    conns: List[TLSConn] = []
    for conn in root.findall("connection"):
        if (conn.get("tl") or "").strip() != tls_id:
            continue
        from_edge = (conn.get("from") or "").strip()
        to_edge = (conn.get("to") or "").strip()
        from_lane = (conn.get("fromLane") or "").strip()
        dir_code = (conn.get("dir") or "").strip()
        if not from_edge or not to_edge or from_edge.startswith(":") or to_edge.startswith(":"):
            continue
        if dir_code not in {"r", "s", "l", "t"}:
            continue
        link_index = conn.get("linkIndex")
        if link_index is None:
            continue
        try:
            link_index_i = int(link_index)
        except Exception:
            continue
        lane_id = f"{from_edge}_{from_lane}"
        lane_dirs.setdefault(lane_id, set()).add(dir_code)
        conns.append(
            TLSConn(
                link_index=link_index_i,
                from_edge=from_edge,
                from_lane=lane_id,
                to_edge=to_edge,
                dir_code=dir_code,
            )
        )

    program = _parse_tls_program(root, tls_id)
    conns.sort(key=lambda c: c.link_index)
    return nodes, edges, lane_dirs, conns, program


def _infer_arm(edge_id: str, edges: Dict[str, Dict[str, object]], nodes: Dict[str, Tuple[float, float]]) -> str:
    info = edges.get(edge_id)
    if not info:
        return "UNK"
    from_node = str(info.get("from") or "")
    to_node = str(info.get("to") or "")
    if from_node not in nodes or to_node not in nodes:
        return "UNK"
    fx, fy = nodes[from_node]
    tx, ty = nodes[to_node]
    dx = tx - fx
    dy = ty - fy
    if abs(dx) >= abs(dy):
        return "W" if dx > 0 else "E"
    return "S" if dy > 0 else "N"


def build_lane_groups(*, net_file: str, tls_id: str, arm_order: Optional[Sequence[str]] = None) -> LaneGroups:
    nodes, edges, lane_dirs, conns, program = _parse_net_tls(net_file=net_file, tls_id=tls_id)

    incoming_edges = sorted({c.from_edge for c in conns})
    outgoing_edges = sorted({c.to_edge for c in conns})
    if not incoming_edges:
        raise RuntimeError(f"No incoming edges found for tls_id={tls_id}")

    edge_to_arm: Dict[str, str] = {}
    for edge_id in incoming_edges:
        edge_to_arm[edge_id] = _infer_arm(edge_id, edges, nodes)

    if arm_order is None:
        arm_order = ["E", "W", "N", "S"]
    arm_order = list(arm_order)

    groups: Dict[str, Dict[str, List[str]]] = {arm: {"TR": [], "LU": []} for arm in arm_order}
    lane_to_group: Dict[str, str] = {}

    for edge_id in incoming_edges:
        arm = edge_to_arm.get(edge_id, "UNK")
        if arm not in groups:
            groups[arm] = {"TR": [], "LU": []}
            if arm not in arm_order:
                arm_order.append(arm)

        lanes = edges.get(edge_id, {}).get("lanes", [])
        if not lanes:
            continue

        tr_lanes: List[str] = []
        lu_lanes: List[str] = []
        for idx, lane_id, _ in lanes:
            dirs = lane_dirs.get(lane_id, set())
            if dirs and dirs.issubset({"s", "r"}):
                tr_lanes.append(lane_id)
            elif dirs and dirs.issubset({"l", "t"}):
                lu_lanes.append(lane_id)
            elif dirs:
                tr_lanes.append(lane_id)
            else:
                if idx == max(l[0] for l in lanes):
                    lu_lanes.append(lane_id)
                else:
                    tr_lanes.append(lane_id)

        if not lu_lanes or not tr_lanes:
            max_idx = max(l[0] for l in lanes)
            tr_lanes = [lane_id for idx, lane_id, _ in lanes if idx != max_idx]
            lu_lanes = [lane_id for idx, lane_id, _ in lanes if idx == max_idx]

        groups[arm]["TR"].extend(tr_lanes)
        groups[arm]["LU"].extend(lu_lanes)
        for lane_id in tr_lanes:
            lane_to_group[lane_id] = f"{arm}_TR"
        for lane_id in lu_lanes:
            lane_to_group[lane_id] = f"{arm}_LU"

    group_order: List[Tuple[str, str]] = []
    for arm in arm_order:
        if arm not in groups:
            continue
        group_order.append((arm, "TR"))
        group_order.append((arm, "LU"))

    return LaneGroups(
        tls_id=tls_id,
        net_file=net_file,
        incoming_edges=incoming_edges,
        outgoing_edges=outgoing_edges,
        arm_order=arm_order,
        edge_to_arm=edge_to_arm,
        groups=groups,
        group_order=group_order,
        lane_to_group=lane_to_group,
        conns=conns,
        program=program,
    )


def served_groups_for_phase(phase_state: str, conns: Sequence[TLSConn], lane_to_group: Dict[str, str]) -> Set[str]:
    served: Set[str] = set()
    for conn in conns:
        if conn.link_index < 0 or conn.link_index >= len(phase_state):
            continue
        if phase_state[conn.link_index] in {"g", "G"}:
            group = lane_to_group.get(conn.from_lane)
            if group:
                served.add(group)
    return served


def _served_connections(phase_state: str, conns: Sequence[TLSConn]) -> List[TLSConn]:
    served: List[TLSConn] = []
    for conn in conns:
        if conn.link_index < 0 or conn.link_index >= len(phase_state):
            continue
        if phase_state[conn.link_index] in {"g", "G"}:
            served.append(conn)
    return served


def verify_phase_semantics(
    lane_groups: LaneGroups,
    action_phase_indices: Sequence[int],
) -> PhaseSemanticsReport:
    issues: List[str] = []
    warnings: List[str] = []
    details: Dict[int, Dict[str, object]] = {}

    phase_states = lane_groups.program.phase_states
    conns = lane_groups.conns

    for phase_index in action_phase_indices:
        if phase_index < 0 or phase_index >= len(phase_states):
            issues.append(f"phase {phase_index} out of range for program length {len(phase_states)}")
            continue
        state = phase_states[int(phase_index)]
        if len(state) != len(conns):
            issues.append(
                f"phase {phase_index} state length {len(state)} != controlled links {len(conns)}"
            )
            continue
        served = _served_connections(state, conns)
        served_dirs = sorted({c.dir_code for c in served})
        served_arms = sorted({lane_groups.edge_to_arm.get(c.from_edge, "UNK") for c in served})
        details[int(phase_index)] = {
            "served_dirs": served_dirs,
            "served_arms": served_arms,
            "served_links": len(served),
        }

    # Expected mapping: A0->phase0 (EW TR), A1->phase2 (NS TR), A2->phase4 (EW LU), A3->phase6 (NS LU)
    expected = {
        0: {"arms": {"E", "W"}, "allow_straight": True, "require_straight": True, "require_right": True},
        2: {"arms": {"N", "S"}, "allow_straight": True, "require_straight": True, "require_right": True},
        4: {"arms": {"E", "W"}, "allow_straight": False, "require_straight": False, "require_right": False},
        6: {"arms": {"N", "S"}, "allow_straight": False, "require_straight": False, "require_right": False},
    }

    available_right: Dict[int, bool] = {}
    available_straight: Dict[int, bool] = {}
    for phase_idx, cfg in expected.items():
        arms = cfg["arms"]
        right_exists = False
        straight_exists = False
        for conn in conns:
            arm = lane_groups.edge_to_arm.get(conn.from_edge, "UNK")
            if arm not in arms:
                continue
            if conn.dir_code == "r":
                right_exists = True
            if conn.dir_code == "s":
                straight_exists = True
        available_right[phase_idx] = right_exists
        available_straight[phase_idx] = straight_exists

    for phase_idx in action_phase_indices:
        if phase_idx not in expected:
            continue
        cfg = expected[int(phase_idx)]
        info = details.get(int(phase_idx))
        if not info:
            continue
        served_dirs = set(info.get("served_dirs", []))
        served_arms = set(info.get("served_arms", []))

        if not cfg["allow_straight"] and "s" in served_dirs:
            issues.append(f"phase {phase_idx} serves straight movements but should not (dir='s')")
        if cfg["require_straight"] and available_straight.get(int(phase_idx), False):
            if "s" not in served_dirs:
                issues.append(f"phase {phase_idx} does not serve any straight movements (dir='s')")
        if cfg["require_right"] and available_right.get(int(phase_idx), False):
            if "r" not in served_dirs:
                warnings.append(f"phase {phase_idx} does not serve any right turns (dir='r')")

        unexpected_arms = sorted([a for a in served_arms if a not in cfg["arms"] and a != "UNK"])
        if unexpected_arms:
            warnings.append(f"phase {phase_idx} serves unexpected arms: {unexpected_arms}")

    ok = len(issues) == 0
    return PhaseSemanticsReport(ok=ok, issues=issues, warnings=warnings, details=details)
