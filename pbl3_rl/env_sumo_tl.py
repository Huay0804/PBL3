import os
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from utils import (  # noqa: E402
    build_sumo_cmd,
    get_controlled_lanes,
    get_lane_features,
    get_phase_served_lanes,
    get_total_queue,
    get_waiting_time_stats,
    resolve_sumo_binary,
    safe_traci_close,
)

import traci  # noqa: E402


class SumoTLEnv:
    def __init__(
        self,
        *,
        sumocfg: str,
        tls_id: str,
        green_phases: Sequence[int],
        yellow_after: Dict[int, int],
        yellow_time: int = 6,
        min_green: int = 15,
        max_green: int = 60,
        max_steps: int = 6000,
        gui: bool = False,
        sumo_binary: Optional[str] = None,
        extra_sumo_args: Optional[Sequence[str]] = None,
    ):
        self.sumocfg = os.path.abspath(sumocfg)
        self.tls_id = tls_id
        self.green_phases = [int(p) for p in green_phases]
        self.yellow_after = {int(k): int(v) for k, v in yellow_after.items()}
        self.yellow_time = int(yellow_time)
        self.min_green = int(min_green)
        self.max_green = int(max_green)
        self.max_steps = int(max_steps)
        self.gui = bool(gui)
        self.sumo_binary = sumo_binary or resolve_sumo_binary(gui=self.gui)
        self.extra_sumo_args = list(extra_sumo_args) if extra_sumo_args else []

        if not self.green_phases:
            raise ValueError("green_phases must be non-empty")
        for phase in self.green_phases:
            if phase not in self.yellow_after:
                raise ValueError(f"yellow_after missing mapping for green phase {phase}")
        if self.min_green <= 0 or self.yellow_time < 0:
            raise ValueError("min_green must be > 0 and yellow_time must be >= 0")
        if self.max_green < self.min_green:
            raise ValueError("max_green must be >= min_green")

        self.controlled_lanes: List[str] = []
        self.controlled_lanes_set: Set[str] = set()
        self.action_served_lanes: List[List[str]] = []

        self._is_open = False
        self._done = False

        self._sim_time = 0
        self._steps = 0
        self._queue_time_sum = 0.0
        self._arrived_total = 0
        self._avg_wait_samples: List[float] = []
        self._max_wait_samples: List[float] = []

        self._current_green_phase: Optional[int] = None
        self._current_green_elapsed = 0

        self._last_obs: Optional[np.ndarray] = None

    @property
    def obs_dim(self) -> int:
        return len(self.controlled_lanes) * 3

    @property
    def num_actions(self) -> int:
        return len(self.green_phases)

    def reset(self, *, seed: int, route_files: Optional[Sequence[str]] = None) -> np.ndarray:
        self.close()

        cmd = build_sumo_cmd(
            sumo_binary=self.sumo_binary,
            sumocfg=self.sumocfg,
            seed=int(seed),
            route_files=route_files,
            max_steps=self.max_steps,
            extra_args=self.extra_sumo_args,
        )

        traci.start(cmd)
        self._is_open = True
        self._done = False

        tls_ids = set(traci.trafficlight.getIDList())
        if self.tls_id not in tls_ids:
            raise RuntimeError(f"tls_id={self.tls_id!r} not found. Available: {', '.join(sorted(tls_ids))}")

        self.controlled_lanes = get_controlled_lanes(self.tls_id)
        if not self.controlled_lanes:
            raise RuntimeError(f"tls_id={self.tls_id!r} has no controlled lanes")
        self.controlled_lanes_set = set(self.controlled_lanes)

        self.action_served_lanes = [
            get_phase_served_lanes(self.tls_id, phase_index) for phase_index in self.green_phases
        ]

        self._sim_time = 0
        self._steps = 0
        self._queue_time_sum = 0.0
        self._arrived_total = 0
        self._avg_wait_samples = []
        self._max_wait_samples = []

        self._current_green_phase = self.green_phases[0]
        self._current_green_elapsed = 0
        self._force_phase(self._current_green_phase)

        self._last_obs = self._get_obs()
        return self._last_obs

    def close(self) -> None:
        if not self._is_open:
            return
        safe_traci_close()
        self._is_open = False
        self._done = False

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        if not self._is_open:
            raise RuntimeError("Environment not started. Call reset() first.")
        if self._done:
            raise RuntimeError("Episode already done. Call reset() to start a new episode.")

        action = int(action)
        if action < 0 or action >= self.num_actions:
            raise ValueError(f"Invalid action={action}. Expected 0..{self.num_actions - 1}")

        requested_green = self.green_phases[action]
        executed_green = requested_green

        if self._current_green_phase is None:
            self._current_green_phase = executed_green
            self._current_green_elapsed = 0
            self._force_phase(executed_green)
        else:
            if executed_green == self._current_green_phase and self._current_green_elapsed >= self.max_green:
                next_idx = (self.green_phases.index(self._current_green_phase) + 1) % self.num_actions
                executed_green = self.green_phases[next_idx]

            if executed_green != self._current_green_phase:
                self._run_yellow(self._current_green_phase)
                self._current_green_phase = executed_green
                self._current_green_elapsed = 0
                self._force_phase(executed_green)

        self._simulate_seconds(self.min_green)
        self._current_green_elapsed += self.min_green

        obs = self._get_obs()
        sum_queue = get_total_queue(self.controlled_lanes)
        avg_wait, max_wait = get_waiting_time_stats(self.controlled_lanes_set)
        self._avg_wait_samples.append(avg_wait)
        self._max_wait_samples.append(max_wait)

        reward = -float(sum_queue) - 0.2 * float(max_wait)

        info: Dict[str, float] = {
            "sim_time": float(self._sim_time),
            "steps": float(self._steps),
            "sum_queue": float(sum_queue),
            "avg_wait": float(avg_wait),
            "max_wait": float(max_wait),
            "throughput": float(self._arrived_total),
            "avg_queue_time": float(self._queue_time_sum / max(1, self._steps)),
            "current_green": float(self._current_green_phase),
            "current_green_elapsed": float(self._current_green_elapsed),
            "requested_green": float(requested_green),
            "executed_green": float(executed_green),
        }

        return obs, reward, self._done, info

    def get_episode_metrics(self) -> Dict[str, float]:
        avg_wait = float(sum(self._avg_wait_samples) / len(self._avg_wait_samples)) if self._avg_wait_samples else 0.0
        max_wait = float(max(self._max_wait_samples)) if self._max_wait_samples else 0.0
        avg_queue_time = float(self._queue_time_sum / max(1, self._steps))
        return {
            "sim_time": float(self._sim_time),
            "steps": float(self._steps),
            "throughput": float(self._arrived_total),
            "avg_queue_time": float(avg_queue_time),
            "avg_wait": float(avg_wait),
            "max_wait": float(max_wait),
        }

    def _get_obs(self) -> np.ndarray:
        queue_lens, veh_counts, mean_speeds = get_lane_features(self.controlled_lanes)
        features: List[float] = []
        for idx in range(len(self.controlled_lanes)):
            features.append(queue_lens[idx])
            features.append(veh_counts[idx])
            features.append(mean_speeds[idx])
        self._last_obs = np.asarray(features, dtype=np.float32)
        return self._last_obs

    def _force_phase(self, phase_index: int) -> None:
        traci.trafficlight.setPhase(self.tls_id, int(phase_index))
        traci.trafficlight.setPhaseDuration(self.tls_id, 999999)

    def _run_yellow(self, from_green_phase: int) -> None:
        yellow_phase = self.yellow_after[int(from_green_phase)]
        self._force_phase(yellow_phase)
        if self.yellow_time > 0:
            self._simulate_seconds(self.yellow_time)

    def _simulate_seconds(self, seconds: int) -> None:
        for _ in range(int(seconds)):
            if self._done:
                break
            traci.simulationStep()
            self._sim_time += 1
            self._steps += 1
            self._arrived_total += int(traci.simulation.getArrivedNumber())
            self._queue_time_sum += get_total_queue(self.controlled_lanes)

            if self._sim_time >= self.max_steps:
                self._done = True
                break
            if traci.simulation.getMinExpectedNumber() <= 0:
                self._done = True
                break
