import tensorflow as tf


def build_dueling_q_model(state_dim: int, num_actions: int, hidden_sizes, lr: float) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(state_dim,), dtype=tf.float32)
    x = inp
    for size in hidden_sizes:
        x = tf.keras.layers.Dense(int(size), activation="relu")(x)

    value = tf.keras.layers.Dense(1, activation="linear")(x)
    advantage = tf.keras.layers.Dense(num_actions, activation="linear")(x)

    def combine_streams(inputs):
        v, a = inputs
        a_mean = tf.reduce_mean(a, axis=1, keepdims=True)
        return v + (a - a_mean)

    q_out = tf.keras.layers.Lambda(combine_streams)([value, advantage])
    model = tf.keras.Model(inp, q_out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)), loss="mse")
    return model
