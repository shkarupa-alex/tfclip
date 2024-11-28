from keras.src import ops


def q_gelu(x, name=None):
    with ops.name_scope(name or "QuickGELU"):
        return x * ops.sigmoid(1.702 * x)
