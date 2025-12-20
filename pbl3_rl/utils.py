import gzip
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _open_text(path: str):
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def _resolve_from(base_file: str, maybe_relative: str) -> str:
    if os.path.isabs(maybe_relative):
        return maybe_relative
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(base_file)), maybe_relative))


def read_sumocfg(sumocfg_path: str) -> Dict[str, object]:
    sumocfg_path = os.path.abspath(sumocfg_path)
    with _open_text(sumocfg_path) as handle:
        tree = ET.parse(handle)
    root = tree.getroot()

    input_node = root.find("input")
    if input_node is None:
        raise RuntimeError("Invalid .sumocfg: missing <input> section")

    def _get_value(tag: str) -> str:
        el = input_node.find(tag)
        return (el.get("value") or "").strip() if el is not None else ""

    net_file = _get_value("net-file")
    route_files = _get_value("route-files")
    additional_files = _get_value("additional-files")

    if not net_file:
        raise RuntimeError("Invalid .sumocfg: missing <net-file value=...>")

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
        "sumocfg": sumocfg_path,
        "net": net_path,
        "routes": route_paths,
        "additional": additional_paths,
    }


def _candidate_sumo_bins() -> List[str]:
    return ["sumo", "sumo.exe", "sumo-gui", "sumo-gui.exe"]


def _is_valid_sumo_home(path: str) -> bool:
    tools_path = os.path.join(path, "tools")
    bin_path = os.path.join(path, "bin")
    traci_path = os.path.join(tools_path, "traci")
    return os.path.isdir(tools_path) and os.path.isdir(bin_path) and os.path.isdir(traci_path)


def _infer_sumo_home_from_path() -> Optional[str]:
    for candidate in _candidate_sumo_bins():
        resolved = shutil.which(candidate)
        if not resolved:
            continue
        return os.path.dirname(os.path.dirname(resolved))
    return None


def _infer_sumo_home_from_common_locations() -> Optional[str]:
    candidates: List[str] = []

    for base in filter(None, [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")]):
        candidates.extend(
            [
                os.path.join(base, "Eclipse", "Sumo"),
                os.path.join(base, "Eclipse", "SUMO"),
                os.path.join(base, "sumo"),
                os.path.join(base, "SUMO"),
            ]
        )

    candidates.extend([r"C:\Sumo", r"C:\SUMO", r"C:\Tools\Sumo", r"C:\Tools\SUMO"])

    home = os.path.expanduser("~")
    candidates.extend([os.path.join(home, "Sumo"), os.path.join(home, "SUMO")])

    seen: Set[str] = set()
    for candidate in candidates:
        candidate = os.path.normpath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isdir(candidate) and _is_valid_sumo_home(candidate):
            return candidate
    return None


def ensure_sumo_tools_in_path() -> str:
    sumo_home = os.environ.get("SUMO_HOME") or _infer_sumo_home_from_path() or _infer_sumo_home_from_common_locations()
    if not sumo_home:
        raise RuntimeError(
            "SUMO not found. Install Eclipse SUMO and set SUMO_HOME (or add SUMO bin folder to PATH)."
        )

    if not _is_valid_sumo_home(sumo_home):
        raise RuntimeError(
            f"Invalid SUMO_HOME={sumo_home!r}: expected folders 'bin' and 'tools/traci'."
        )

    tools_path = os.path.join(sumo_home, "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)

    os.environ["SUMO_HOME"] = sumo_home
    return sumo_home


def resolve_sumo_binary(gui: bool = False) -> str:
    sumo_home = ensure_sumo_tools_in_path()
    candidates = ["sumo-gui.exe", "sumo-gui"] if gui else ["sumo.exe", "sumo"]

    for name in candidates:
        candidate = os.path.join(sumo_home, "bin", name)
        if os.path.isfile(candidate):
            return candidate

    for name in candidates:
        resolved = shutil.which(name)
        if resolved:
            return resolved

    kind = "sumo-gui" if gui else "sumo"
    raise RuntimeError(f"Could not find {kind} binary. Set SUMO_HOME correctly or add SUMO 'bin' to PATH.")


ensure_sumo_tools_in_path()
import traci  # noqa: E402


def unique_sorted(values: Iterable[str]) -> List[str]:
    return sorted(set(values))


def get_controlled_lanes(tls_id: str) -> List[str]:
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    lanes = [lane for lane in lanes if lane and not lane.startswith(":")]
    return unique_sorted(lanes)


def get_tls_phase_states(tls_id: str) -> List[str]:
    program_id = traci.trafficlight.getProgram(tls_id)
    logics = traci.trafficlight.getAllProgramLogics(tls_id)
    for logic in logics:
        pid = getattr(logic, "programID", None) or getattr(logic, "programId", None)
        if pid == program_id:
            phases = getattr(logic, "phases", [])
            return [getattr(phase, "state", "") for phase in phases]

    phases = getattr(logics[0], "phases", []) if logics else []
    return [getattr(phase, "state", "") for phase in phases]


def get_phase_served_lanes(tls_id: str, phase_index: int) -> List[str]:
    states = get_tls_phase_states(tls_id)
    if phase_index < 0 or phase_index >= len(states):
        raise RuntimeError(f"phase_index={phase_index} out of range (phases={len(states)})")

    state = states[phase_index]
    controlled_links = traci.trafficlight.getControlledLinks(tls_id)
    if len(state) != len(controlled_links):
        raise RuntimeError(
            f"TLS state length mismatch: len(state)={len(state)} len(controlled_links)={len(controlled_links)}"
        )

    served: Set[str] = set()
    for idx, sig in enumerate(state):
        if sig not in ("G", "g"):
            continue
        for conn in controlled_links[idx]:
            from_lane = conn[0]
            if from_lane and not from_lane.startswith(":"):
                served.add(from_lane)

    return sorted(served)


def get_lane_features(lanes: Sequence[str]) -> Tuple[List[float], List[float], List[float]]:
    queue_lens: List[float] = []
    veh_counts: List[float] = []
    mean_speeds: List[float] = []
    for lane in lanes:
        queue_lens.append(float(traci.lane.getLastStepHaltingNumber(lane)))
        veh_counts.append(float(traci.lane.getLastStepVehicleNumber(lane)))
        mean_speeds.append(float(traci.lane.getLastStepMeanSpeed(lane)))
    return queue_lens, veh_counts, mean_speeds


def get_total_queue(lanes: Sequence[str]) -> float:
    return float(sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes))


def get_waiting_time_stats(lanes_set: Set[str]) -> Tuple[float, float]:
    waits: List[float] = []
    for veh_id in traci.vehicle.getIDList():
        try:
            lane_id = traci.vehicle.getLaneID(veh_id)
        except traci.exceptions.TraCIException:
            continue
        if lane_id in lanes_set:
            waits.append(float(traci.vehicle.getAccumulatedWaitingTime(veh_id)))

    if not waits:
        return 0.0, 0.0

    avg_wait = float(sum(waits) / len(waits))
    max_wait = float(max(waits))
    return avg_wait, max_wait


def safe_traci_close() -> None:
    try:
        traci.close(False)
    except Exception:
        return


def build_sumo_cmd(
    *,
    sumo_binary: str,
    sumocfg: str,
    seed: int,
    route_files: Optional[Sequence[str]] = None,
    additional_files: Optional[Sequence[str]] = None,
    max_steps: Optional[int] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> List[str]:
    cmd: List[str] = [
        sumo_binary,
        "-c",
        os.path.abspath(sumocfg),
        "--seed",
        str(int(seed)),
        "--no-step-log",
        "true",
        "--duration-log.disable",
        "true",
    ]

    if max_steps is not None:
        cmd.extend(["--waiting-time-memory", str(int(max_steps))])

    if route_files:
        cmd.extend(["--route-files", ",".join([os.path.abspath(p) for p in route_files])])
    if additional_files:
        cmd.extend(["--additional-files", ",".join([os.path.abspath(p) for p in additional_files])])
    if extra_args:
        cmd.extend(list(extra_args))
    return cmd


def generate_random_trips(
    *,
    net_file: str,
    out_trip_file: str,
    seed: int,
    end_time: int,
    insertion_density: float = 12.0,
    fringe_factor: float = 5.0,
    min_distance: float = 300.0,
    vclass: str = "passenger",
    prefix: str = "veh",
    trip_attributes: str = 'departLane="best" departSpeed="max"',
    validate: bool = True,
    remove_loops: bool = True,
    lanes: bool = True,
) -> str:
    sumo_home = ensure_sumo_tools_in_path()
    random_trips = os.path.join(sumo_home, "tools", "randomTrips.py")
    if not os.path.isfile(random_trips):
        raise RuntimeError(f"randomTrips.py not found at {random_trips}")

    os.makedirs(os.path.dirname(os.path.abspath(out_trip_file)), exist_ok=True)

    cmd: List[str] = [
        sys.executable,
        random_trips,
        "-n",
        os.path.abspath(net_file),
        "-o",
        os.path.abspath(out_trip_file),
        "-b",
        "0",
        "-e",
        str(int(end_time)),
        "--seed",
        str(int(seed)),
        "--insertion-density",
        str(float(insertion_density)),
        "--fringe-factor",
        str(float(fringe_factor)),
        "--vehicle-class",
        vclass,
        "--prefix",
        prefix,
        "--min-distance",
        str(float(min_distance)),
        "--trip-attributes",
        trip_attributes,
    ]

    if lanes:
        cmd.append("--lanes")
    if validate:
        cmd.append("--validate")
    if remove_loops:
        cmd.append("--remove-loops")

    subprocess.run(cmd, check=True)
    return os.path.abspath(out_trip_file)

