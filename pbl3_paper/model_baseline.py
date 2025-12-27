import tensorflow as tf


def build_q_model(state_dim: int, num_actions: int, hidden_sizes, lr: float) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(state_dim,), dtype=tf.float32)
    x = inp
    hidden_sizes = [int(s) for s in hidden_sizes]
    if len(hidden_sizes) != 2:
        raise ValueError("Baseline DQN expects exactly 2 hidden layers.")
    # Match DQL-TSC: first hidden layer linear (no activation), second ReLU.
    x = tf.keras.layers.Dense(hidden_sizes[0])(x)
    x = tf.keras.layers.Dense(hidden_sizes[1], activation="relu")(x)
    out = tf.keras.layers.Dense(num_actions, activation="linear")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)), loss="mse")
    return model
