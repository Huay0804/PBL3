import os
import shutil
import sys
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PBL3_ROOT = os.path.normpath(os.path.join(THIS_DIR, ".."))
for p in (THIS_DIR, PBL3_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from pbl3_shared.sumo_lane_cells import (  # noqa: E402
    LaneGroups,
    PhaseSemanticsReport,
    build_lane_groups,
    read_sumocfg,
    served_groups_for_phase,
)


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


class SumoTLEnvCells:
    def __init__(
        self,
        *,
        sumocfg: str,
        tls_id: str,
        seed: int = 0,
        gui: bool = False,
        sim_seconds: int = 5400,
        num_cells: int = 10,
        action_phase_indices: Sequence[int],
        timing_mode: str,
        green_step: int,
        yellow_time: int,
        route_files: Optional[Sequence[str]] = None,
        extra_sumo_args: Optional[Sequence[str]] = None,
    ):
        self.sumocfg = os.path.abspath(sumocfg)
        self.tls_id = str(tls_id)
        self.seed = int(seed)
        self.gui = bool(gui)
        self.sim_seconds = int(sim_seconds)
        self.num_cells = int(num_cells)
        self.route_files = list(route_files) if route_files else None
        self.extra_sumo_args = list(extra_sumo_args) if extra_sumo_args else None

        if self.num_cells != 10:
            raise ValueError("This lane-based environment currently supports num_cells=10 only.")

        cfg = read_sumocfg(self.sumocfg)
        self.net_file = str(cfg["net"])
        self._lane_groups: LaneGroups = build_lane_groups(net_file=self.net_file, tls_id=self.tls_id)

        self.green_phases = [int(p) for p in action_phase_indices]
        self.yellow_after = {int(p): int(p) + 1 for p in self.green_phases}

        self.timing_mode = str(timing_mode).upper().strip()
        if self.timing_mode not in {"STRICT_MATCH_README", "KEEP_TLS_NATIVE"}:
            raise RuntimeError(f"Unknown timing mode: {self.timing_mode}")

        self.green_step = int(green_step)
        self.yellow_time = int(yellow_time)

        for gp in self.green_phases:
            if gp not in self.yellow_after:
                raise RuntimeError(f"Missing yellow_after mapping for green phase {gp}")
            yp = int(self.yellow_after[gp])
            if yp < 0 or yp >= len(self._lane_groups.program.phase_states):
                raise RuntimeError(f"yellow phase index {yp} out of range for program length")

        self._sumo_binary = resolve_sumo_binary(gui=self.gui)
        self.group_order = list(self._lane_groups.group_order)
        self.state_dim = len(self.group_order) * self.num_cells
        self.num_actions = len(self.green_phases)
        self.action_names = [f"PHASE{p}" for p in self.green_phases]

        self._incoming_lanes = sorted(
            {lane for arm in self._lane_groups.groups.values() for grp in arm.values() for lane in grp}
        )
        self._lane_lengths: Dict[str, float] = {}

        self._done = False
        self._sim_time = 0
        self._steps = 0
        self._curr_action: Optional[int] = None
        self._w_t = 0.0
        self._vqs = 0.0
        self._sum_queue = 0.0

        self._throughput_junction = 0
        self._seen_incoming: Set[str] = set()
        self._seen_throughput: Set[str] = set()

    def close(self) -> None:
        safe_traci_close()

    def reset(self, *, seed: Optional[int] = None, route_files: Optional[Sequence[str]] = None) -> np.ndarray:
        safe_traci_close()
        if seed is not None:
            self.seed = int(seed)
        if route_files is not None:
            self.route_files = list(route_files) if route_files else None

        cmd = self._build_sumo_cmd()
        scenario_dir = os.path.dirname(self.sumocfg)
        with pushd(scenario_dir):
            traci.start(cmd)

        self._sim_time = 0
        self._steps = 0
        self._done = False
        self._curr_action = None
        self._w_t = 0.0
        self._vqs = 0.0
        self._sum_queue = 0.0
        self._throughput_junction = 0
        self._seen_incoming = set()
        self._seen_throughput = set()

        self._init_lanes()
        self._w_t = float(self._get_waiting_time())
        return self._encode_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        if self._done:
            return self._encode_state(), 0.0, True, self.info()

        action = int(action)
        if action < 0 or action >= self.num_actions:
            raise RuntimeError(f"action out of range: {action}")

        if self._curr_action is None:
            self._curr_action = action
        elif action != self._curr_action:
            self._apply_yellow(self._curr_action)
            self._curr_action = action

        self._apply_green(self._curr_action)

        w_prev = float(self._w_t)
        self._w_t = float(self._get_waiting_time())
        reward = 0.9 * w_prev - self._w_t

        self._sum_queue = float(self._sum_queue_snapshot())
        self._vqs += float(self._sum_queue)

        obs = self._encode_state()
        return obs, float(reward), bool(self._done), self.info()

    def info(self) -> Dict[str, float]:
        return {
            "time": float(self._sim_time),
            "w_t": float(self._w_t),
            "sum_queue": float(self._sum_queue),
            "vqs": float(self._vqs),
            "throughput_junction": float(self._throughput_junction),
            "action": float(self._curr_action if self._curr_action is not None else -1),
            "green_phase": float(self.green_phases[self._curr_action]) if self._curr_action is not None else -1.0,
        }

    def served_groups_by_phase(self) -> Dict[int, Set[str]]:
        out: Dict[int, Set[str]] = {}
        for ph in self.green_phases:
            state = self._lane_groups.program.phase_states[int(ph)]
            out[int(ph)] = served_groups_for_phase(state, self._lane_groups.conns, self._lane_groups.lane_to_group)
        return out

    def verify_phase_semantics_live(self) -> PhaseSemanticsReport:
        controlled_links = traci.trafficlight.getControlledLinks(self.tls_id)
        issues: List[str] = []
        warnings: List[str] = []
        details: Dict[int, Dict[str, object]] = {}

        if not controlled_links:
            issues.append("No controlled links from TraCI for this tls_id.")
            return PhaseSemanticsReport(ok=False, issues=issues, warnings=warnings, details=details)

        dir_map: Dict[Tuple[str, str], str] = {}
        for conn in self._lane_groups.conns:
            dir_map[(conn.from_lane, conn.to_edge)] = conn.dir_code

        phase_states = self._lane_groups.program.phase_states

        for phase_index in self.green_phases:
            if phase_index < 0 or phase_index >= len(phase_states):
                issues.append(f"phase {phase_index} out of range for program length {len(phase_states)}")
                continue
            state = phase_states[int(phase_index)]
            if len(state) != len(controlled_links):
                issues.append(
                    f"phase {phase_index} state length {len(state)} != controlled links {len(controlled_links)}"
                )
                continue

            served_dirs: Set[str] = set()
            served_arms: Set[str] = set()
            served_links = 0
            for i, sig in enumerate(state):
                if sig not in {"g", "G"}:
                    continue
                for conn in controlled_links[i]:
                    if not conn or len(conn) < 2:
                        continue
                    from_lane = str(conn[0])
                    to_lane = str(conn[1])
                    if from_lane.startswith(":") or to_lane.startswith(":"):
                        continue
                    to_edge = to_lane.rsplit("_", 1)[0]
                    dir_code = dir_map.get((from_lane, to_edge), "?")
                    from_edge = from_lane.rsplit("_", 1)[0]
                    arm = self._lane_groups.edge_to_arm.get(from_edge, "UNK")
                    served_dirs.add(dir_code)
                    served_arms.add(arm)
                    served_links += 1

            if "?" in served_dirs:
                served_dirs.discard("?")
                warnings.append(f"phase {phase_index} has links with unknown dir code")

            details[int(phase_index)] = {
                "served_dirs": sorted(served_dirs),
                "served_arms": sorted(served_arms),
                "served_links": served_links,
            }

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
            for conn in self._lane_groups.conns:
                arm = self._lane_groups.edge_to_arm.get(conn.from_edge, "UNK")
                if arm not in arms:
                    continue
                if conn.dir_code == "r":
                    right_exists = True
                if conn.dir_code == "s":
                    straight_exists = True
            available_right[phase_idx] = right_exists
            available_straight[phase_idx] = straight_exists

        for phase_idx in self.green_phases:
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
            str(int(self.sim_seconds)),
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

    def _init_lanes(self) -> None:
        tls_ids = set(traci.trafficlight.getIDList())
        if self.tls_id not in tls_ids:
            raise RuntimeError(f"tls_id={self.tls_id!r} not present in running simulation.")
        self._lane_lengths = {lane: float(traci.lane.getLength(lane)) for lane in self._incoming_lanes}

    def _distance_to_cell(self, distance_to_tls: float, lane_length: float) -> int:
        if lane_length <= 0:
            return self.num_cells - 1
        bin_size = lane_length / float(self.num_cells)
        if bin_size <= 0:
            return self.num_cells - 1
        idx = int(distance_to_tls / bin_size)
        if idx < 0:
            idx = 0
        if idx >= self.num_cells:
            idx = self.num_cells - 1
        return idx

    def _encode_state(self) -> np.ndarray:
        state = np.zeros(self.state_dim, dtype=np.float32)
        for group_idx, (arm, group) in enumerate(self.group_order):
            lanes = self._lane_groups.groups.get(arm, {}).get(group, [])
            if not lanes:
                continue
            for lane_id in lanes:
                lane_length = float(self._lane_lengths.get(lane_id, 0.0))
                if lane_length <= 0.0:
                    continue
                for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
                    try:
                        lane_pos = float(traci.vehicle.getLanePosition(veh_id))
                    except Exception:
                        continue
                    dist = max(0.0, lane_length - lane_pos)
                    cell = self._distance_to_cell(dist, lane_length)
                    state[group_idx * self.num_cells + cell] = 1.0
        return state

    def _get_waiting_time(self) -> float:
        total = 0.0
        incoming_set = set(self._incoming_lanes)
        for veh_id in traci.vehicle.getIDList():
            try:
                lane_id = traci.vehicle.getLaneID(veh_id)
            except Exception:
                continue
            if lane_id in incoming_set:
                try:
                    total += float(traci.vehicle.getAccumulatedWaitingTime(veh_id))
                except Exception:
                    continue
        return float(total)

    def _sum_queue_snapshot(self) -> float:
        q = 0.0
        for lane in self._incoming_lanes:
            try:
                q += float(traci.lane.getLastStepHaltingNumber(lane))
            except Exception:
                continue
        return float(q)

    def _apply_green(self, action: int) -> None:
        phase = int(self.green_phases[int(action)])
        traci.trafficlight.setPhase(self.tls_id, phase)
        traci.trafficlight.setPhaseDuration(self.tls_id, 999999)
        self._simulate_seconds(self._green_duration_for_phase(phase))

    def _apply_yellow(self, action: int) -> None:
        phase = int(self.green_phases[int(action)])
        yellow_phase = int(self.yellow_after[phase])
        traci.trafficlight.setPhase(self.tls_id, yellow_phase)
        traci.trafficlight.setPhaseDuration(self.tls_id, 999999)
        self._simulate_seconds(self._yellow_duration_for_phase(phase))

    def _green_duration_for_phase(self, phase: int) -> int:
        if self.timing_mode == "KEEP_TLS_NATIVE":
            return max(1, int(round(self._lane_groups.program.phase_durations[int(phase)])))
        return max(1, int(self.green_step))

    def _yellow_duration_for_phase(self, phase: int) -> int:
        if self.timing_mode == "KEEP_TLS_NATIVE":
            yp = int(self.yellow_after[int(phase)])
            return max(1, int(round(self._lane_groups.program.phase_durations[yp])))
        return max(1, int(self.yellow_time))

    def _simulate_seconds(self, seconds: int) -> None:
        for _ in range(int(seconds)):
            if self._done:
                break
            traci.simulationStep()
            self._sim_time = float(traci.simulation.getTime())
            self._mark_throughput()
            self._steps += 1
            if self._steps >= int(self.sim_seconds):
                self._done = True
            elif traci.simulation.getMinExpectedNumber() <= 0 and self._steps > 10:
                self._done = True

    def _mark_throughput(self) -> None:
        incoming_edges = set(self._lane_groups.incoming_edges)
        outgoing_edges = set(self._lane_groups.outgoing_edges)
        for veh_id in traci.vehicle.getIDList():
            try:
                road_id = traci.vehicle.getRoadID(veh_id)
            except Exception:
                continue
            if road_id in incoming_edges:
                self._seen_incoming.add(veh_id)
            if road_id in outgoing_edges and veh_id in self._seen_incoming and veh_id not in self._seen_throughput:
                self._seen_throughput.add(veh_id)
                self._throughput_junction += 1
