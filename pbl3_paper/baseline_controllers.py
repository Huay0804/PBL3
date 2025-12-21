from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from env_sumo_tl import SumoTLEnv  # noqa: E402

import traci  # noqa: E402


@dataclass
class FixedTimePolicy:
    cycle: Sequence[int] = (0, 1)

    def __post_init__(self) -> None:
        self._idx = 0

    def reset(self) -> None:
        self._idx = 0

    def act(self, _obs: np.ndarray, _info: dict) -> int:
        action = int(self.cycle[self._idx])
        self._idx = (self._idx + 1) % len(self.cycle)
        return action


class MaxQueuePolicy:
    """
    Heuristic baseline (stronger than fixed-time):
      - infer served lanes for each green phase from the TLS program state string,
      - choose the action whose served lanes have the largest queue (halting vehicles).
    """

    def __init__(self, env: SumoTLEnv, *, tie_break_keep_last: bool = True):
        self.env = env
        self.tie_break_keep_last = bool(tie_break_keep_last)
        self._last_action = 0
        self._served_lanes_by_action: Optional[List[List[str]]] = None

    def reset(self) -> None:
        self._last_action = 0
        self._served_lanes_by_action = self._infer_served_lanes()

    def act(self, _obs: np.ndarray, _info: dict) -> int:
        scores = self._compute_scores()
        best = int(np.argmax(scores))
        if self.tie_break_keep_last and scores[best] == scores[self._last_action]:
            best = int(self._last_action)
        self._last_action = best
        return best

    def _compute_scores(self) -> List[int]:
        if not self._served_lanes_by_action:
            self._served_lanes_by_action = self._infer_served_lanes()

        scores: List[int] = []
        for lanes in self._served_lanes_by_action:
            q = 0
            for lane in lanes:
                try:
                    q += int(traci.lane.getLastStepHaltingNumber(lane))
                except traci.exceptions.TraCIException:
                    continue
            scores.append(q)
        return scores

    def _infer_served_lanes(self) -> List[List[str]]:
        phase_states = _get_tls_phase_states(self.env.tls_id)
        served: List[List[str]] = []
        for green_phase in self.env.green_phases:
            served.append(_served_lanes_for_phase(self.env.tls_id, int(green_phase), phase_states))
        return served


def _get_tls_phase_states(tls_id: str) -> List[str]:
    program_id = traci.trafficlight.getProgram(tls_id)
    logics = traci.trafficlight.getAllProgramLogics(tls_id)
    for logic in logics:
        pid = getattr(logic, "programID", None) or getattr(logic, "programId", None)
        if pid == program_id:
            phases = getattr(logic, "phases", []) or []
            return [getattr(p, "state", "") for p in phases]
    phases = getattr(logics[0], "phases", []) if logics else []
    return [getattr(p, "state", "") for p in phases]


def _served_lanes_for_phase(tls_id: str, phase_index: int, phase_states: Sequence[str]) -> List[str]:
    if phase_index < 0 or phase_index >= len(phase_states):
        raise RuntimeError(f"phase_index={phase_index} out of range (phases={len(phase_states)})")

    state = phase_states[phase_index]
    controlled_links = traci.trafficlight.getControlledLinks(tls_id)
    if len(state) != len(controlled_links):
        raise RuntimeError(
            f"TLS state length mismatch: len(state)={len(state)} len(controlled_links)={len(controlled_links)}"
        )

    served_set = set()
    for idx, sig in enumerate(state):
        if sig not in ("G", "g"):
            continue
        for conn in controlled_links[idx]:
            from_lane = conn[0]
            if from_lane and not str(from_lane).startswith(":"):
                served_set.add(str(from_lane))
    return sorted(served_set)


def run_episode(env: SumoTLEnv, policy, *, max_decisions: Optional[int] = None) -> dict:
    obs = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()

    done = False
    episode_reward = 0.0
    sum_neg_reward = 0.0
    sum_intersection_queue = 0.0
    decisions = 0
    info = env.info()

    while not done:
        action = int(policy.act(obs, info))
        obs, r, done, info = env.step(action)
        episode_reward += float(r)
        if float(r) < 0.0:
            sum_neg_reward += float(r)
        sum_intersection_queue += float(info.get("sum_queue", 0.0))
        decisions += 1
        if max_decisions is not None and decisions >= int(max_decisions):
            break

    out = dict(info)
    out["episode_reward"] = float(episode_reward)
    out["sum_neg_reward"] = float(sum_neg_reward)
    out["nwt_abs"] = float(abs(sum_neg_reward))
    out["sum_intersection_queue"] = float(sum_intersection_queue)
    out["decisions"] = float(decisions)
    return out
