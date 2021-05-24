# Notebook-style Python script, compatible with VS Code's Python plugin.

# %%
import tensorflow as tf

# %%
batch = tf.constant(
    [  # Batch dimension
        [  # Sequence 1
            [0.1, 0.1],  # Token 1
            [0.2, 0.2],  # Token 2
            [0.3, 0.3],  # Token 3
        ],
    ]
)

# %%
# Disable bias to make manual loop slightly easier to implement
rnn = tf.keras.layers.SimpleRNN(1, use_bias=False)
# This needs to run before we can access the inner workings of the layer.
res = rnn(batch)
print(f"Simple RNN kernel: {rnn.weights[0]}")
print(f"Simple RNN recurrent kernel: {rnn.weights[1]}")
print()
print(f"Simple RNN output: {res}")

# %%
def simple_rnn_loop(batch, rnn: tf.keras.layers.SimpleRNN):
    # Code is inspired by the SimpleRNNCell call()
    # https://github.com/tensorflow/tensorflow/blob/8309456f10f01b31c1eb965f971080121d86c705/tensorflow/python/keras/layers/recurrent.py#L1376-L1396
    state = rnn.get_initial_state(batch)[0]
    kernel = rnn.weights[0]
    recurrent_kernel = rnn.weights[1]

    for timestep in range(batch.shape[1]):
        inp = batch[:, timestep, :]
        # Kernel input for timestep
        h = tf.keras.backend.dot(inp, kernel)
        # Output (new state) for timestep
        o = h + tf.keras.backend.dot(state, recurrent_kernel)
        o = rnn.activation(o)
        state = o

        print(f"Input timestep: {inp}")
        print(f"Kernel input: {h}")
        print(f"Next state: {state}")
        print()

    print(f"Final output: {o}")


# The final output should correspond to the output of the simple RNN.
simple_rnn_loop(batch, rnn)

# %%
# Disable bias to make manual loop slightly easier to implement
gru = tf.keras.layers.GRU(1, use_bias=False)
res = gru(batch)
print(f"GRU kernel: {gru.weights[0]}")
print(f"GRU recurrent kernel: {gru.weights[1]}")
print()
print(f"GRU output: {res}")

# %%
def gru_rnn_loop(batch, gru: tf.keras.layers.GRU):
    # Code is inspired by the GRUCell call():
    # https://github.com/tensorflow/tensorflow/blob/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/tensorflow/python/keras/layers/recurrent.py#L1829-L1932
    state = gru.get_initial_state(batch)[0]
    kernel = gru.weights[0]
    recurrent_kernel = gru.weights[1]

    for timestep in range(batch.shape[1]):
        inp = batch[:, timestep, :]
        matrix_x = tf.keras.backend.dot(inp, kernel)
        x_z, x_r, x_h = tf.split(matrix_x, 3, axis=-1)
        matrix_inner = tf.keras.backend.dot(state, recurrent_kernel)
        recurrent_z, recurrent_r, recurrent_h = tf.split(
            matrix_inner, [gru.units, gru.units, -1], axis=-1
        )

        # Recurrent activations for update and reset gate
        z = gru.recurrent_activation(x_z + recurrent_z)
        r = gru.recurrent_activation(x_r + recurrent_r)

        # Apply reset gate to hidden state
        recurrent_h = r * recurrent_h

        # Candidate state
        hh = gru.activation(x_h + recurrent_h)

        # Previous and candidate state mixed by update gate
        o = z * state + (1 - z) * hh
        state = o

        print(f"Input timestep: {inp}")
        print(f"Kernel input: x_z: {x_z}, x_r: {x_r}, x_h: {x_h}")
        print(f"Gates: z: {z}, r: {r}")
        print(f"Candidate state: {hh}")
        print(f"Next state: {state}")
        print()

    print(f"Final output: {o}")


# The final output should correspond to the output of the GRU layer.
gru_rnn_loop(batch, gru)

# %%
lstm = tf.keras.layers.LSTM(2, use_bias=False)
res = lstm(batch)
print(f"LSTM kernel: {lstm.weights[0]}")
print(f"LSTM recurrent kernel: {lstm.weights[1]}")
print()
print(f"LSTM output: {res}")

# %%
def lstm_rnn_loop(batch, lstm: tf.keras.layers.GRU):
    # Code is inspired by the LSTMCell call():
    # https://github.com/tensorflow/tensorflow/blob/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/tensorflow/python/keras/layers/recurrent.py#L2414-L2472

    # Previous and carry states
    h, c = lstm.get_initial_state(batch)
    kernel = lstm.weights[0]
    recurrent_kernel = lstm.weights[1]

    for timestep in range(batch.shape[1]):
        inp = batch[:, timestep, :]

        z = tf.keras.backend.dot(inp, kernel)
        z += tf.keras.backend.dot(h, recurrent_kernel)

        z = tf.split(z, num_or_size_splits=4, axis=1)

        z0, z1, z2, z3 = z
        i = lstm.recurrent_activation(z0)
        f = lstm.recurrent_activation(z1)
        c = f * c + i * lstm.activation(z2)
        o = lstm.recurrent_activation(z3)

        h = o * lstm.activation(c)
        o = h

        print(f"Input timestep: {inp}")
        print(f"Gates: z: {z}")
        print(f"Next state: c: {c}, h: {h}")
        print()

    print(f"Final output: {o}")


lstm_rnn_loop(batch, lstm)
# %%
