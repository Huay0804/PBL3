import os
from typing import Tuple

import numpy as np
import tensorflow as tf

try:
    from pbl3_shared.sumo_lane_cells import load_experiment_config
except Exception:
    load_experiment_config = None

from pbl3_paper.model_baseline import build_q_model


def _load_config() -> Tuple[int, int, float]:
    state_dim = 80
    num_actions = 4
    gamma = 0.75
    if load_experiment_config is None:
        return state_dim, num_actions, gamma
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "experiment_config.yaml")
    try:
        cfg = load_experiment_config(os.path.abspath(cfg_path))
    except Exception:
        return state_dim, num_actions, gamma
    actions = cfg.get("actions", {})
    dqn = cfg.get("dqn", {})
    phase_indices = actions.get("action_phase_indices", [0, 2, 4, 6])
    return state_dim, int(len(phase_indices)), float(dqn.get("gamma", gamma))


def test_forward_shape() -> None:
    state_dim, num_actions, _ = _load_config()
    model = build_q_model(state_dim, num_actions, hidden_sizes=[400, 400], lr=1e-3)
    batch = np.zeros((4, state_dim), dtype=np.float32)
    q = model.predict(batch, verbose=0)
    print("T1 forward Q shape:", q.shape)


def test_td_target() -> None:
    _, _, gamma = _load_config()
    q_target_next = np.array(
        [
            [1.0, 2.0, 3.0, 0.0],
            [0.5, 0.2, 0.1, 0.4],
        ],
        dtype=np.float32,
    )
    rewards = np.array([1.0, 2.0], dtype=np.float32)
    dones = np.array([0.0, 1.0], dtype=np.float32)
    max_next = np.max(q_target_next, axis=1)
    targets = rewards + (1.0 - dones) * gamma * max_next
    print("T2 targets:", targets)


def test_action_gradient() -> None:
    state_dim, num_actions, _ = _load_config()
    model = build_q_model(state_dim, num_actions, hidden_sizes=[400, 400], lr=1e-3)
    states = tf.zeros((1, state_dim), dtype=tf.float32)
    actions = tf.constant([2], dtype=tf.int32)
    targets = tf.constant([1.0], dtype=tf.float32)
    with tf.GradientTape() as tape:
        q_pred = model(states, training=True)
        tape.watch(q_pred)
        q_taken = tf.gather(q_pred, actions, axis=1, batch_dims=1)
        loss = tf.reduce_mean(tf.square(targets - q_taken))
    grads = tape.gradient(loss, q_pred)
    print("T3 grad w.r.t Q:", grads.numpy())


def main() -> int:
    test_forward_shape()
    test_td_target()
    test_action_gradient()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
