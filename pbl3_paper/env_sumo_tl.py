import os
import shutil
import sys
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


def ensure_sumo_tools_in_path() -> str:
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        for candidate in ("sumo", "sumo.exe", "sumo-gui", "sumo-gui.exe"):
            resolved = shutil.which(candidate)
            if resolved:
                sumo_home = os.path.dirname(os.path.dirname(resolved))
                break
    if not sumo_home:
        raise RuntimeError(
            "SUMO not found. Install Eclipse SUMO and set SUMO_HOME (or add SUMO/bin to PATH)."
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
    raise RuntimeError(f"Could not find {kind} binary. Set SUMO_HOME correctly or add SUMO/bin to PATH.")


ensure_sumo_tools_in_path()
import traci  # noqa: E402


@contextmanager
def pushd(path: str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def safe_traci_close() -> None:
    try:
        traci.close(False)
    except Exception:
        return


def lane_to_edge(lane_id: str) -> str:
    return lane_id.rsplit("_", 1)[0]


def _resolve_from(base_file: str, maybe_relative: str) -> str:
    if os.path.isabs(maybe_relative):
        return maybe_relative
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(base_file)), maybe_relative))


def read_sumocfg(sumocfg_path: str) -> Dict[str, object]:
    sumocfg_path = os.path.abspath(sumocfg_path)
    tree = ET.parse(sumocfg_path)
    root = tree.getroot()
    input_node = root.find("input")
    if input_node is None:
        raise RuntimeError("Invalid .sumocfg: missing <input>")

    def _val(tag: str) -> str:
        el = input_node.find(tag)
        return (el.get("value") or "").strip() if el is not None else ""

    net_rel = _val("net-file")
    if not net_rel:
        raise RuntimeError("Invalid .sumocfg: missing <net-file value=...>")

    route_rel = _val("route-files")
    add_rel = _val("additional-files")

    net_path = _resolve_from(sumocfg_path, net_rel)
    routes = [_resolve_from(sumocfg_path, p.strip()) for p in route_rel.split(",") if p.strip()]
    additional = [_resolve_from(sumocfg_path, p.strip()) for p in add_rel.split(",") if p.strip()]

    return {"sumocfg": sumocfg_path, "net": net_path, "routes": routes, "additional": additional}


@dataclass(frozen=True)
class TLSConn:
    link_index: int
    from_edge: str
    to_edge: str
    dir_code: str  # r / s / l / t


@dataclass(frozen=True)
class TLSProgram:
    phase_states: List[str]
    phase_durations: List[float]


def parse_tls_from_net(*, net_file: str, tls_id: str) -> Tuple[List[TLSConn], TLSProgram]:
    tree = ET.parse(net_file)
    root = tree.getroot()

    phase_states: List[str] = []
    phase_durations: List[float] = []
    candidates = [t for t in root.findall("tlLogic") if (t.get("id") or "") == tls_id]
    if not candidates:
        raise RuntimeError(f"tls_id={tls_id!r} not found in net file: {net_file}")
    tl = next((t for t in candidates if (t.get("programID") or t.get("programId") or "") == "0"), candidates[0])

    for ph in tl.findall("phase"):
        state = (ph.get("state") or "").strip()
        dur = float(ph.get("duration") or 0.0)
        phase_states.append(state)
        phase_durations.append(dur)

    conns: Dict[int, TLSConn] = {}
    for conn in root.findall("connection"):
        if (conn.get("tl") or "") != tls_id:
            continue
        link_index_raw = (conn.get("linkIndex") or "").strip()
        if not link_index_raw:
            continue
        try:
            link_index = int(link_index_raw)
        except ValueError:
            continue

        from_edge = (conn.get("from") or "").strip()
        to_edge = (conn.get("to") or "").strip()
        dir_code = (conn.get("dir") or "").strip()
        if not from_edge or not to_edge or from_edge.startswith(":") or to_edge.startswith(":"):
            continue
        if dir_code not in {"r", "s", "l", "t"}:
            continue
        if link_index in conns:
            continue
        conns[link_index] = TLSConn(link_index=link_index, from_edge=from_edge, to_edge=to_edge, dir_code=dir_code)

    if not conns:
        raise RuntimeError(f"No <connection tl=... linkIndex=...> found for tls_id={tls_id!r} in {net_file}")

    ordered = [conns[i] for i in sorted(conns.keys())]
    return ordered, TLSProgram(phase_states=phase_states, phase_durations=phase_durations)


def validate_phase_mapping(*, program: TLSProgram, green_phases: Sequence[int], yellow_after: Dict[int, int]) -> None:
    n = len(program.phase_states)
    if n <= 0:
        raise RuntimeError("TLS program has no phases.")

    for gp in green_phases:
        if gp < 0 or gp >= n:
            raise RuntimeError(f"green phase index {gp} out of range (phases={n})")
        if gp not in yellow_after:
            raise RuntimeError(f"Missing yellow_after mapping for green phase {gp}")
        yp = int(yellow_after[gp])
        if yp < 0 or yp >= n:
            raise RuntimeError(f"yellow phase index {yp} out of range (phases={n})")


def infer_main_green_phases(*, program: TLSProgram, k: int = 2) -> List[int]:
    candidates: List[Tuple[float, int, int]] = []
    for i, state in enumerate(program.phase_states):
        if "y" in state.lower():
            continue
        gcount = sum(1 for c in state if c in ("G", "g"))
        if gcount <= 0:
            continue
        dur = program.phase_durations[i] if i < len(program.phase_durations) else 0.0
        candidates.append((float(dur), int(gcount), int(i)))

    candidates.sort(reverse=True)
    if len(candidates) < int(k):
        raise RuntimeError(f"Not enough green phases to infer {k} main actions. Found {len(candidates)} candidates.")

    phases = sorted([c[2] for c in candidates[: int(k)]])
    return phases


def infer_yellow_after(*, program: TLSProgram, green_phases: Sequence[int]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for gp in green_phases:
        yp = int(gp) + 1
        if yp < len(program.phase_states) and "y" in program.phase_states[yp].lower():
            mapping[int(gp)] = int(yp)
            continue
        raise RuntimeError(
            f"Could not infer yellow phase after green {gp}. "
            "Pass yellow_after explicitly or ensure yellow is the next phase index."
        )
    return mapping


class SumoTLEnv:
    ACTION_NAMES: Tuple[str, ...] = ()

    def __init__(
        self,
        *,
        sumocfg: str,
        tls_id: str,
        seed: int = 0,
        gui: bool = False,
        max_steps: int = 5400,
        green_duration: int = 33,
        yellow_duration: int = 6,
        num_cells: int = 10,
        green_phases: Optional[Sequence[int]] = None,
        yellow_after: Optional[Dict[int, int]] = None,
        route_files: Optional[Sequence[str]] = None,
        extra_sumo_args: Optional[Sequence[str]] = None,
    ):
        self.sumocfg = os.path.abspath(sumocfg)
        self.tls_id = str(tls_id)
        self.seed = int(seed)
        self.gui = bool(gui)
        self.max_steps = int(max_steps)
        self.green_duration = int(green_duration)
        self.yellow_duration = int(yellow_duration)
        self.num_cells = int(num_cells)
        self.route_files = list(route_files) if route_files else None
        self.extra_sumo_args = list(extra_sumo_args) if extra_sumo_args else None

        if self.num_cells != 10:
            raise ValueError("This paper-mode environment currently supports num_cells=10 only.")

        cfg = read_sumocfg(self.sumocfg)
        self.net_file = str(cfg["net"])

        self._conns, self._program = parse_tls_from_net(net_file=self.net_file, tls_id=self.tls_id)

        self.incoming_edges: List[str] = sorted({c.from_edge for c in self._conns})
        self.outgoing_edges: List[str] = sorted({c.to_edge for c in self._conns})

        self.green_phases = [int(p) for p in green_phases] if green_phases is not None else infer_main_green_phases(program=self._program, k=2)
        self.yellow_after = dict(yellow_after) if yellow_after is not None else infer_yellow_after(program=self._program, green_phases=self.green_phases)
        validate_phase_mapping(program=self._program, green_phases=self.green_phases, yellow_after=self.yellow_after)

        self.ACTION_NAMES = tuple([f"PHASE{p}" for p in self.green_phases])

        self._incoming_edges_order = sorted(self.incoming_edges)
        self._edge_pos = {e: i for i, e in enumerate(self._incoming_edges_order)}

        # Lane-based state: 1 group per incoming edge, 10 cells per arm.
        self.state_dim = len(self._incoming_edges_order) * self.num_cells
        self.num_actions = len(self.green_phases)

        self._base_lane_length = 500.0
        self._base_cell_bounds = [7.0, 14.0, 21.0, 28.0, 40.0, 60.0, 100.0, 200.0, 350.0, 500.0]

        self._sumo_binary = resolve_sumo_binary(gui=self.gui)

        self._lane_lengths: Dict[str, float] = {}
        self._incoming_lanes: List[str] = []

        self._done = False
        self._sim_time = 0
        self._steps = 0
        self._curr_action: Optional[int] = None
        self._curr_wait_time = 0.0

        self._seen_incoming: Set[str] = set()
        self._seen_throughput: Set[str] = set()
        self._throughput_junction = 0

        self._departed_total = 0
        self._arrived_total = 0
        self._last_departed = 0
        self._last_arrived = 0
        self._last_running = 0

        self._last_sum_queue = 0.0
        self._last_sum_wait = 0.0
        self._last_mean_speed_in = 0.0

        self._seconds = 0
        self._sum_queue_time = 0.0
        self._sum_wait_time = 0.0
        self._mean_speed_time = 0.0

    def close(self) -> None:
        safe_traci_close()

    def reset(self, *, seed: Optional[int] = None, route_files: Optional[Sequence[str]] = None) -> np.ndarray:
        safe_traci_close()

        if seed is not None:
            self.seed = int(seed)
        if route_files is not None:
            self.route_files = list(route_files)

        self._done = False
        self._sim_time = 0
        self._steps = 0
        self._curr_action = None
        self._curr_wait_time = 0.0

        self._seen_incoming = set()
        self._seen_throughput = set()
        self._throughput_junction = 0

        self._departed_total = 0
        self._arrived_total = 0
        self._last_departed = 0
        self._last_arrived = 0
        self._last_running = 0

        self._last_sum_queue = 0.0
        self._last_sum_wait = 0.0
        self._last_mean_speed_in = 0.0

        self._seconds = 0
        self._sum_queue_time = 0.0
        self._sum_wait_time = 0.0
        self._mean_speed_time = 0.0

        cmd = self._build_sumo_cmd()

        with pushd(os.path.dirname(self.sumocfg)):
            traci.start(cmd)

        self._init_lanes()
        # Start with action 0 (first green phase).
        self._curr_action = 0
        self._set_tls_phase(self.green_phases[self._curr_action])
        self._update_last_metrics()
        return self._encode_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        if self._done:
            return self._encode_state(), 0.0, True, self.info()

        action = int(action)
        if action < 0 or action >= self.num_actions:
            raise ValueError(f"Invalid action={action}. Expected [0..{self.num_actions-1}]")

        old_action = self._curr_action
        if old_action is None:
            old_action = action
            self._curr_action = action

        if action != old_action:
            old_green = int(self.green_phases[int(old_action)])
            yellow_phase = int(self.yellow_after[old_green])
            self._set_tls_phase(yellow_phase)
            if self.yellow_duration > 0:
                self._simulate_seconds(self.yellow_duration)
            self._curr_action = action

        self._set_tls_phase(self.green_phases[int(self._curr_action)])
        self._simulate_seconds(self.green_duration)

        new_wait = float(self._get_waiting_time())
        reward = 0.9 * float(self._curr_wait_time) - new_wait
        self._curr_wait_time = new_wait

        obs = self._encode_state()
        return obs, float(reward), bool(self._done), self.info()

    def info(self) -> Dict[str, float]:
        avg_queue = (self._sum_queue_time / self._seconds) if self._seconds > 0 else 0.0
        avg_wait = (self._sum_wait_time / self._seconds) if self._seconds > 0 else 0.0
        mean_speed = (self._mean_speed_time / self._seconds) if self._seconds > 0 else 0.0
        return {
            "time": float(self._sim_time),
            "sum_queue": float(self._last_sum_queue),
            "sum_wait": float(self._last_sum_wait),
            "mean_speed_in": float(self._last_mean_speed_in),
            "throughput_junction": float(self._throughput_junction),
            "departed": float(self._last_departed),
            "arrived": float(self._last_arrived),
            "running": float(self._last_running),
            "departed_total": float(self._departed_total),
            "arrived_total": float(self._arrived_total),
            "avg_queue": float(avg_queue),
            "avg_wait": float(avg_wait),
            "avg_speed_in": float(mean_speed),
            "action": float(self._curr_action if self._curr_action is not None else -1),
            "green_phase": float(self.green_phases[self._curr_action]) if self._curr_action is not None else -1.0,
        }

    def _build_sumo_cmd(self) -> List[str]:
        cmd: List[str] = [
            self._sumo_binary,
            "-c",
            self.sumocfg,
            "--seed",
            str(int(self.seed)),
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
            "--waiting-time-memory",
            str(int(self.max_steps)),
            "--tripinfo-output",
            "NUL",
            "--statistic-output",
            "NUL",
        ]
        if self.route_files:
            cmd.extend(["--route-files", ",".join([os.path.abspath(p) for p in self.route_files])])
        if self.extra_sumo_args:
            cmd.extend(list(self.extra_sumo_args))
        return cmd

    def _set_tls_phase(self, phase_index: int) -> None:
        traci.trafficlight.setPhase(self.tls_id, int(phase_index))
        traci.trafficlight.setPhaseDuration(self.tls_id, 999999)

    def _simulate_seconds(self, seconds: int) -> None:
        for _ in range(int(seconds)):
            if self._done:
                break
            self._tick()

    def _mark_seen_incoming(self) -> None:
        for edge_id in self.incoming_edges:
            for veh_id in traci.edge.getLastStepVehicleIDs(edge_id):
                self._seen_incoming.add(veh_id)

    def _mark_throughput(self) -> None:
        for edge_id in self.outgoing_edges:
            for veh_id in traci.edge.getLastStepVehicleIDs(edge_id):
                if veh_id not in self._seen_incoming:
                    continue
                if veh_id in self._seen_throughput:
                    continue
                self._seen_throughput.add(veh_id)
                self._throughput_junction += 1

    def _mean_speed_in(self) -> float:
        weighted = 0.0
        total = 0.0
        for lane in self._incoming_lanes:
            n = float(traci.lane.getLastStepVehicleNumber(lane))
            v = float(traci.lane.getLastStepMeanSpeed(lane))
            weighted += n * v
            total += n
        return (weighted / total) if total > 0 else 0.0

    def _update_last_metrics(self) -> None:
        sum_queue = float(sum(traci.lane.getLastStepHaltingNumber(lane) for lane in self._incoming_lanes))
        sum_wait = float(sum(float(traci.lane.getWaitingTime(lane)) for lane in self._incoming_lanes))
        ms = float(self._mean_speed_in())
        self._last_sum_queue = sum_queue
        self._last_sum_wait = sum_wait
        self._last_mean_speed_in = ms

    def _tick(self) -> None:
        self._mark_seen_incoming()
        traci.simulationStep()
        self._sim_time = int(traci.simulation.getTime())

        departed = int(traci.simulation.getDepartedNumber())
        arrived = int(traci.simulation.getArrivedNumber())
        running = int(traci.simulation.getMinExpectedNumber())
        self._departed_total += departed
        self._arrived_total += arrived

        self._last_departed = departed
        self._last_arrived = arrived
        self._last_running = running

        self._mark_seen_incoming()
        self._mark_throughput()
        self._update_last_metrics()

        self._seconds += 1
        self._sum_queue_time += self._last_sum_queue
        self._sum_wait_time += self._last_sum_wait
        self._mean_speed_time += self._last_mean_speed_in

        self._steps += 1
        if self._steps >= int(self.max_steps):
            self._done = True
        elif running <= 0 and self._steps > 10:
            self._done = True

    def _init_lanes(self) -> None:
        tls_ids = set(traci.trafficlight.getIDList())
        if self.tls_id not in tls_ids:
            raise RuntimeError(f"tls_id={self.tls_id!r} not present in running simulation.")

        incoming_set = set(self.incoming_edges)
        lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        lanes = [l for l in lanes if l and not l.startswith(":") and lane_to_edge(l) in incoming_set]
        self._incoming_lanes = sorted(set(lanes))
        if not self._incoming_lanes:
            raise RuntimeError(
                f"No incoming controlled lanes found for tls_id={self.tls_id!r}. incoming_edges={self.incoming_edges}"
            )

        self._lane_lengths = {lane: float(traci.lane.getLength(lane)) for lane in self._incoming_lanes}

    def _distance_to_cell(self, distance_to_tls: float, lane_length: float) -> int:
        scale = lane_length / self._base_lane_length if lane_length > 0 else 1.0
        bounds = [b * scale for b in self._base_cell_bounds]
        for idx, bound in enumerate(bounds):
            if distance_to_tls < bound:
                return idx
        return self.num_cells - 1

    def _encode_state(self) -> np.ndarray:
        state = np.zeros(self.state_dim, dtype=np.float32)
        for edge in self.incoming_edges:
            for veh_id in traci.edge.getLastStepVehicleIDs(edge):
                try:
                    lane_id = traci.vehicle.getLaneID(veh_id)
                    lane_pos = float(traci.vehicle.getLanePosition(veh_id))
                    lane_length = float(self._lane_lengths.get(lane_id, 0.0))
                    if lane_length <= 0.0:
                        continue
                except traci.exceptions.TraCIException:
                    continue

                edge_id = edge
                edge_pos = self._edge_pos.get(edge_id)
                if edge_pos is None:
                    continue

                distance_to_tls = max(0.0, lane_length - lane_pos)
                cell = self._distance_to_cell(distance_to_tls, lane_length)
                state[edge_pos * self.num_cells + cell] = 1.0

        return state

    def _get_waiting_time(self) -> float:
        total = 0.0
        for edge in self.incoming_edges:
            for veh_id in traci.edge.getLastStepVehicleIDs(edge):
                try:
                    total += float(traci.vehicle.getAccumulatedWaitingTime(veh_id))
                except traci.exceptions.TraCIException:
                    continue
        return total
